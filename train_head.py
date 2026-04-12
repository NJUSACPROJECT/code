import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'  
import torch
import copy
import numpy as np
import argparse
from models import ClusterHead
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from eval_utils import cluster_metric
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset,Dataset
import faiss
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torchvision import datasets, transforms
from loss_utils import entropy,consistency_loss,num_consistency_loss,DataContrastiveLoss
from data_utils import NeighborsDataset, mine_nearest_neighbors,TestDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 对某些CUDA操作是必须的

# 使用方法（在程序开始时调用）
set_seed(42)  # 可以任意设置喜欢的种子数

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, mode='min'):
        """
        :param patience: 当验证指标在多少个 epoch 内没有提升时停止训练
        :param verbose: 是否打印早停信息
        :param delta: 判定指标提升的最小变化量
        :param mode: 'min' 表示指标越小越好，'max' 表示指标越大越好
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.best_model_state = None

        if mode == 'min':
            self.val_metric_best = float('inf')
        else:
            self.val_metric_best = float('-inf')

    def __call__(self, val_metric, model):
        if self.mode == 'min':
            score = -val_metric
        else:
            score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif (self.mode == 'min' and score > self.best_score + self.delta) or (self.mode == 'max' and score < self.best_score - self.delta):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        if self.verbose:
            if self.mode == 'min':
                print(f'Validation metric decreased ({self.val_metric_best:.6f} --> {val_metric:.6f}).  Saving model...')
            else:
                print(f'Validation metric increased ({self.val_metric_best:.6f} --> {val_metric:.6f}).  Saving model...')
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_metric_best = val_metric

def infer(model, dataloader):
    model.eval()
    preds = []
    logits_image = []
    with torch.no_grad():
        for iter, (text,image) in enumerate(dataloader):
            text=text[0].cuda()
            image = image[0].cuda()
            logit_text, logit_image= model(text, image)#[batchsize cluter_num]
            pred = torch.argmax(logit_image, dim=1).cpu().numpy()
            preds.append(pred)
            logits_image.append(logit_image.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    logits_image = np.concatenate(logits_image, axis=0)
    return preds, logits_image

if __name__ == "__main__":
    print("running...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--consist_coeff', type=float, default=0.6,
                        help='coefficient for consistency loss')
    parser.add_argument('--entropy_coeff', type=float, default=-5.0,
                        help='coefficient for entropy loss')
    parser.add_argument('--neighbor_numbers', type=int, default=50,
                        help='coefficient for entropy loss')
    parser.add_argument('--noise_rate', type=float, default=0.7)
    parser.add_argument('--dataset', type=str, default="CIFAR-20")
    parser.add_argument('--dataset_name', type=str, default="cifar20")#imagenetdogs
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='alpha')
    parser.add_argument('--add_noise', type=bool, default=False)
    parser.add_argument('--topk', type=int, default=50)
    args = parser.parse_args()

    #改数据集
    dataset = args.dataset
    dataset_name=args.dataset_name
   
    use_checkpoint=False
    add_noise=args.add_noise
 
    consist_coeff = args.consist_coeff   # 初始值 0.5
    entropy_coeff=args.entropy_coeff
    temperature =  torch.nn.Parameter(torch.tensor(1.1))               # 初始值 1
    alpha=args.alpha
    noise_rate=args.noise_rate

    # 超参数
    epochs =30
    batch_size = 512
    topK = args.topk

    if dataset == "CIFAR-10" or dataset == "STL-10" or dataset == "ImageNet-10":
        cluster_num = 10
    elif dataset == "CIFAR-20":#实际是cifar100 用粗粒度标签
        cluster_num = 20
    elif dataset == "ImageNet-Dogs":
        cluster_num = 15
    else:
        raise NotImplementedError
    #改数据集
    index_file_train= dataset_name+"_semantic_index.index"
    nouns_embedding = np.load(dataset_name+"_sberttext_embedding_train.npy")

    nouns_embedding = nouns_embedding / np.linalg.norm(
        nouns_embedding, axis=1, keepdims=True
    )
    

    
    labels_train = np.loadtxt("./data/" + dataset + "_labels_train.txt").astype(np.int64)

    images_embedding_train = np.load("./data/" + dataset + "_image_embedding_train.npy")
    images_embedding_train = images_embedding_train / np.linalg.norm(
        images_embedding_train, axis=1, keepdims=True
    )

    images_embedding_test = np.load("./data/" + dataset + "_image_embedding_test.npy")
    images_embedding_test = images_embedding_test / np.linalg.norm(
        images_embedding_test, axis=1, keepdims=True
    )
    labels_test = np.loadtxt("./data/" + dataset + "_labels_test.txt")
    labels_train = np.loadtxt("./data/" + dataset + "_labels_train.txt")

    
    model = ClusterHead(in_dim=512,text_in_dim=768,num_clusters=cluster_num).cuda()
    if use_checkpoint==True:
        state_dict = torch.load('./outputs/best_model_'+dataset_name+'(1).pth')
        model.load_state_dict( state_dict)
    

    dataset_text_train = TensorDataset(torch.from_numpy(nouns_embedding).float())
    dataset_image_train = TensorDataset(
        torch.from_numpy(images_embedding_train).float()
    )
    dataset_image_test = TensorDataset(torch.from_numpy(images_embedding_test).float())
    # dataset_text_test = TensorDataset(torch.from_numpy(nouns_embedding_test).float())
  
    indices_text = mine_nearest_neighbors(nouns_embedding, topk=topK,index_file=None)
    indices_image = mine_nearest_neighbors(images_embedding_train, topk=topK)
        
   
    dataset = NeighborsDataset(
        dataset_text_train, dataset_image_train, indices_text, indices_image,k=args.neighbor_numbers
    )

    dataset_train= TestDataset(
        dataset_image_train, dataset_image_train
    )####邻居没改所以是乱的 注意这里有没有文本
    dataloader_traininfer=DataLoader(
        dataset_train, batch_size=batch_size, shuffle=False, drop_last=False
    )
    dataloader_train = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True  
    )
    dataset_test = TestDataset(
        dataset_image_test,dataset_image_test
    )####邻居没改所以是乱的 注意这里有没有文本
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, drop_last=False  
    )
                        
    DC_loss=DataContrastiveLoss( temperature=temperature,max_epoch=epochs)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99))
    patience = 5  # 早停的耐心值
    early_stopping = EarlyStopping(patience=patience, verbose=True, mode='min')
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {param_count / 1e6:.2f} M")
    # 估算显存占用（假设float32，每个参数4字节）
    # print(f"模型参数显存占用: {param_count * 4 / 1024**3:.2f} GB")
    
    print("Start infer...")
    preds,logit_feature = infer(model, dataloader_test)
    cluster_metric(labels_test, preds)
    print("Start training...")
    beta=args.beta


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99),)

    for epoch in range(epochs):
        all_features=[]
        epoch_losses = []
        total_loss = 0
        model.eval()
        all_losses = np.zeros(len(dataloader_train.dataset))
       
        
        model.train()
        loss_consist_epoch = loss_entropy_epoch = loss_epoch=loss_dc_epoch =0.0
        for iter, (indices, text, image, neigh_text, neigh_image,image2text,text2image) in enumerate(dataloader_train):
            indices = indices.cuda()
            text = text[0].cuda()
            image = image[0].cuda()#torch.Size([512, 512]) 

            neigh_text = neigh_text[0].cuda()
            neigh_image = neigh_image[0].cuda()# torch.Size([512, 5, 512])
            image2text = image2text[0].cuda()
            text2image = text2image[0].cuda()
            logit_text, logit_image= model(text, image)
            neigh_logit_text, neigh_logit_image= model(neigh_text, neigh_image)
            logit_image2text,logit_text2image=model(image2text,text2image)#torch.Size([512, 5, 10])
            if epoch < epochs // 2:
                scaling_factor = 1.0
            else:
                scaling_factor = 2.0
            k=scaling_factor
           
            loss_consist =beta* consistency_loss(logit_text, logit_image)+(1-beta)*num_consistency_loss(logit_text, logit_image)

            # + consistency_loss(logit_text, logit_image)
            loss_entropy = entropy(logit_text) + entropy(logit_image)
           
            ifweight=True
            # alpha=torch.sigmoid(alpha).cuda()trytry-----------------------------------------------

            loss_dc1=DC_loss(logit_image, logit_text2image, logit_text,logit_image2text,logit_image,logit_text2image,k, ifweight,alpha)            
            loss_dc2=DC_loss(logit_image, neigh_logit_text,logit_text, neigh_logit_image, logit_image, neigh_logit_text,k, ifweight,alpha)
            +DC_loss(logit_text, neigh_logit_image, logit_image, neigh_logit_text,logit_text, neigh_logit_image,k, ifweight,alpha)

            loss_dc=loss_dc1+loss_dc2
            loss =entropy_coeff*loss_entropy+consist_coeff*loss_consist+loss_dc
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss_distill_epoch += loss_dc.item()
            loss_consist_epoch += loss_consist.item()
            loss_entropy_epoch += loss_entropy.item()
            loss_dc_epoch += loss_dc.item()

            if (iter + 1) % 50 == 0 or iter + 1 == len(dataloader_train):
                print(
                    "[Epoch {}/{}] [Iter {}/{}] Loss dc: {:.4f} Loss Consist: {:.4f} Loss Entropy: {:.4f} ".format(
                        epoch + 1,
                        epochs,
                        iter + 1,
                        len(dataloader_train),
                        loss_dc.item(),
                        loss_consist.item(),
                        loss_entropy.item(),
                       
                    )
                )
        print(
            "[Epoch: {}] Loss dc: {:.4f} Loss Consist: {:.4f} Loss Entropy: {:.4f} Loss: {:.4f} ".format(
                epoch + 1,
                loss_dc_epoch/ (iter + 1),
                loss_consist_epoch / (iter + 1),
                loss_entropy_epoch / (iter + 1),
                loss / (iter + 1),
            )
        )
        # 评估模型
        preds,logit_feature = infer(model, dataloader_test)
        total_loss += loss.item()
        acc=cluster_metric(labels_test, preds)
        early_stopping(acc, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    if early_stopping.best_model_state is not None:
        torch.save(early_stopping.best_model_state, 'best_model_'+dataset_name+'.pth')
        model.load_state_dict(early_stopping.best_model_state)
        preds,logit_feature = infer(model, dataloader_test)
        cluster_metric(labels_test, preds)
      
        print("ifweight: ",ifweight)
        if add_noise is True:
            print("noise rate: ", noise_rate)
        print("entropy_coeff: ",entropy_coeff,"\n consist_coeff: ",consist_coeff,"\n temperature: ",temperature, "\n alpha: ",alpha)
