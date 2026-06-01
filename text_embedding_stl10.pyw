import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from torchvision import datasets
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BlipImageProcessor, Blip2Processor, Blip2ForConditionalGeneration

# 读取原始图片喂给 BLIP-2
train_dataset = datasets.STL10(root='./data', split='train', download=True)

print("🚀 正在加载 BLIP-2 视觉大模型...")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).cuda()

descriptions = []
for i in tqdm(range(len(train_dataset)), desc="BLIP-2 正在看 STL-10..."):
    image, _ = train_dataset[i]
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
    
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True).replace("Answer:", "").strip()
    
    if not description: description = "a photo of an object"
    descriptions.append(description)

print("🚀 正在将文本描述转化为 768 维 SBERT 特征...")
sbert = SentenceTransformer("all-mpnet-base-v2")
embeddings = sbert.encode(descriptions, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()

np.save("stl10_sberttext_embedding_train.npy", embeddings)
print(f"✅ 文本特征保存成功！Shape: {embeddings.shape}")