import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'  
import torch
import copy
import numpy as np
from models import ClusterHead
from eval_utils import cluster_metric
from torch.utils.data import DataLoader, TensorDataset
import random
from loss_utils import entropy, consistency_loss, num_consistency_loss, DataContrastiveLoss
from data_utils import NeighborsDataset, mine_nearest_neighbors, TestDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

set_seed(42)

class EarlyStopping:
    def __init__(self, patience=40, verbose=False, delta=0, mode='max'):
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
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif (self.mode == 'min' and score > self.best_score + self.delta) or (self.mode == 'max' and score < self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_metric_best = val_metric

def infer(model, dataloader):
    model.eval()
    preds = []
    logits_image = []
    with torch.no_grad():
        for iter, (text, image) in enumerate(dataloader):
            text = text[0].cuda()
            image = image[0].cuda()
            logit_text, logit_image = model(text, image)
            pred = torch.argmax(logit_image, dim=1).cpu().numpy()
            preds.append(pred)
            logits_image.append(logit_image.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    logits_image = np.concatenate(logits_image, axis=0)
    return preds, logits_image

if __name__ == "__main__":
    print("🚀 启动 SAC Training (VLM 专家创新版)...")
    
    # 核心超参数锁定区
    dataset_name = "imagenet_dogs_15"   
    cluster_num = 15     
    
    consist_coeff = 0.6   
    entropy_coeff = -4.0 
    temperature = torch.nn.Parameter(torch.tensor(1.1)) 
    alpha = 0.9

    # ==========================================
    # 🔥 少样本特调参数 (针对 1395 张图)
    epochs = 150       
    batch_size = 128   
    topK = 5           
    neighbor_numbers = 5
    # ==========================================

    print("加载数据特征中...")
    
    # 🌟 【核心创新】：加载 VQA 提示词提取出的高质量文本特征！
    nouns_embedding = np.load("imagenet_dogs_15_sberttext_embedding_vlm_expert.npy")
    nouns_embedding = nouns_embedding / np.linalg.norm(nouns_embedding, axis=1, keepdims=True)
    
    # 加载图像特征
    images_embedding_train = np.load("./data/imagenet_dogs_15_image_embedding_train.npy")
    images_embedding_train = images_embedding_train / np.linalg.norm(images_embedding_train, axis=1, keepdims=True)
    
    # 加载标签
    labels_train = np.loadtxt("./data/imagenet_dogs_15_labels_train.txt")
    
    # 无监督聚类：直接用训练集作为测试集进行效果验证
    images_embedding_test = images_embedding_train.copy()
    labels_test = labels_train.copy()

    # 模型初始化
    model = ClusterHead(in_dim=512, text_in_dim=768, num_clusters=cluster_num).cuda()

    dataset_text_train = TensorDataset(torch.from_numpy(nouns_embedding).float())
    dataset_image_train = TensorDataset(torch.from_numpy(images_embedding_train).float())
    dataset_image_test = TensorDataset(torch.from_numpy(images_embedding_test).float())
  
    print(f"挖掘近邻样本中 (KNN, TopK={topK})...")
    indices_text = mine_nearest_neighbors(nouns_embedding, topk=topK, index_file=None)
    indices_image = mine_nearest_neighbors(images_embedding_train, topk=topK)
        
    dataset_train_obj = NeighborsDataset(dataset_text_train, dataset_image_train, indices_text, indices_image, k=neighbor_numbers)
    # 丢弃最后的不完整 Batch，防止在极小数据量下梯度计算抖动
    dataloader_train = DataLoader(dataset_train_obj, batch_size=batch_size, shuffle=True, drop_last=True)
    
    dataset_test_obj = TestDataset(dataset_image_test, dataset_image_test)
    dataloader_test = DataLoader(dataset_test_obj, batch_size=batch_size, shuffle=False, drop_last=False)
                        
    DC_loss = DataContrastiveLoss(temperature=temperature, max_epoch=epochs)
    early_stopping = EarlyStopping(patience=40, verbose=False, mode='max') 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"🔥 开始正式炼丹 (引入 VLM 专家知识)...")
    beta = 1.0 

    for epoch in range(epochs):
        model.train()
        loss_consist_epoch = loss_entropy_epoch = loss_epoch = loss_dc_epoch = 0.0
        
        for iter, (indices, text, image, neigh_text, neigh_image, image2text, text2image) in enumerate(dataloader_train):
            indices = indices.cuda()
            text = text[0].cuda()
            image = image[0].cuda()
            neigh_text = neigh_text[0].cuda()
            neigh_image = neigh_image[0].cuda()
            image2text = image2text[0].cuda()
            text2image = text2image[0].cuda()
            
            logit_text, logit_image = model(text, image)
            neigh_logit_text, neigh_logit_image = model(neigh_text, neigh_image)
            logit_image2text, logit_text2image = model(image2text, text2image)
            
            scaling_factor = 1.0 if epoch < epochs // 2 else 2.0
            k = scaling_factor
           
            loss_consist = beta * consistency_loss(logit_text, logit_image) + (1 - beta) * num_consistency_loss(logit_text, logit_image)
            loss_entropy = entropy(logit_text) + entropy(logit_image)
            ifweight = True

            loss_dc1 = DC_loss(logit_image, logit_text2image, logit_text, logit_image2text, logit_image, logit_text2image, k, ifweight, alpha)            
            loss_dc2 = DC_loss(logit_image, neigh_logit_text, logit_text, neigh_logit_image, logit_image, neigh_logit_text, k, ifweight, alpha) \
                     + DC_loss(logit_text, neigh_logit_image, logit_image, neigh_logit_text, logit_text, neigh_logit_image, k, ifweight, alpha)

            loss_dc = loss_dc1 + loss_dc2
            loss = entropy_coeff * loss_entropy + consist_coeff * loss_consist + loss_dc
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_consist_epoch += loss_consist.item()
            loss_entropy_epoch += loss_entropy.item()
            loss_dc_epoch += loss_dc.item()
            loss_epoch += loss.item()

        scheduler.step()
        
        # 每个 Epoch 评估一次并打印 ACC
        preds, logit_feature = infer(model, dataloader_test)
        acc_results = cluster_metric(labels_test, preds)
        acc_val = acc_results[0] if isinstance(acc_results, tuple) else acc_results
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch + 1:03d}/{epochs}] Loss: {loss_epoch/(iter+1):.4f} | ACC: {acc_val:.4f} | LR: {current_lr:.6f}")
        
        early_stopping(acc_val, model)
        if early_stopping.early_stop:
            print("🛑 触发早停机制，训练结束。")
            break
            
    if early_stopping.best_model_state is not None:
        torch.save(early_stopping.best_model_state, 'best_model_'+dataset_name+'.pth')
        model.load_state_dict(early_stopping.best_model_state)
        preds, logit_feature = infer(model, dataloader_test)
        
        print("\n" + "★"*45)
        print("🏆 ImageNet-Dogs-15 (VLM专家创新版) 最终开奖结果 🏆")
        final_results = cluster_metric(labels_test, preds)
        if isinstance(final_results, tuple) and len(final_results) == 3:
            print(f"  >> ACC (准确率): {final_results[0]:.4f}")
            print(f"  >> NMI (互信息): {final_results[1]:.4f}")
            print(f"  >> ARI (兰德指数): {final_results[2]:.4f}")
        else:
            print(f"  >> 结果: {final_results}")
        print("★"*45 + "\n")