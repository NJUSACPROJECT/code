import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BlipImageProcessor, Blip2Processor, Blip2ForConditionalGeneration

# 1. 修复致命Bug：这里必须是 CIFAR100，否则文本和图像就是跨服聊天！
transform = transforms.Compose([
    transforms.ToTensor(),
])
cifar100_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# 2. 加载 BLIP-2
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

descriptions = []
dataset = cifar100_dataset

for i in tqdm(range(len(dataset)), desc="BLIP-2 Image Captioning..."):
    image, _ = dataset[i]
    image = transforms.ToPILImage()(image)
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    
    if "Answer:" in description:
        answer_start = description.find("Answer:") + len("Answer:")
        answer = description[answer_start:].strip()
    else:
        answer = description
        
    if answer.strip() == "":
        print(f"Warning: Empty description for image {i}")

    descriptions.append(answer)

# 保存文本描述
with open("cifar20_train_descriptions_blip2.txt", "w") as f:
    for description in descriptions:
        f.write(description + "\n")
print("Descriptions saved to 'cifar20_train_descriptions_blip2.txt'")


# 3. 提取文本特征 (SBERT)
class Config:
    text_file = "cifar20_train_descriptions_blip2.txt"
    sbert_model = "all-mpnet-base-v2" 
    save_file = "cifar20_sberttext_embedding_train.npy"

config = Config()

class DATAPROCESSING(Dataset):
    def __init__(self, train=True):
        with open(config.text_file) as f:
            self.texts = [line.strip() for line in f]
        print("text length: ", len(self.texts))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def build_semantic(dataset):
    sbert = SentenceTransformer(config.sbert_model)
    embeddings = []
    for texts in tqdm(DataLoader(dataset, batch_size=512), desc="SBERT TEXT EMBEDDING..."):
        emb = sbert.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    np.save(config.save_file, embeddings)
    print("Semantic Embeddings saved. Shape:", embeddings.shape)

train_set = DATAPROCESSING(train=True)
build_semantic(train_set)