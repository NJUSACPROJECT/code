
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration



#genetate sentence use cifar20 as example

transform = transforms.Compose([
    transforms.ToTensor(),  # image to tensor[0, 1]
])
cifar100_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# BLIP-2 
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# if use  prompt
# prompt = "Question: How to describe the main content of the image in a way that is suitable for clustering? Answer:"
# prompt="Question: Describe the main content of the image in detail? Answer:"

descriptions = []
dataset=cifar100_dataset

for i in tqdm(range(len(dataset))):
    image, _ = dataset[i]
    
    # tensor to PIL
    image = transforms.ToPILImage()(image)
    # use prompt 
    # inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    # not use prompt 
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    # DESCRIPTION
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    
    # DELETE "Answer:"
    if "Answer:" in description:
        answer_start = description.find("Answer:") + len("Answer:")
        answer = description[answer_start:].strip()
    else:
        answer = description
    if answer.strip() == "":
        print(f"Warning: Empty description for image {i}")


    descriptions.append(answer)
print(len(descriptions))

# save
with open("cifar100_train_descriptions_blip2.txt", "w") as f:
    for description in descriptions:
        f.write(description+ "\n")

print("Descriptions saved to 'cifar100_train_descriptions_blip2.txt'")

# embedding sentence
class Config:
    text_file = "cifar100_train_descriptions_blip2.txt"
    sbert_model = "all-mpnet-base-v2" 
    index_file="cifar20_semantic_index.index"
    save_file="cifar20_sberttext_embedding_train_blip.npy"

config = Config()

class DATAPROCESSING(Dataset):
    def __init__(self, train=True):
      
        
        with open(config.text_file) as f:
            self.texts = [line.strip() for line in f]
        print("text length: ",len(self.texts))


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text

def build_semantic(dataset):
    sbert = SentenceTransformer(config.sbert_model)

    embeddings = []
    for texts in tqdm(DataLoader(dataset, batch_size=512), desc="TEXT EMBEDDING..."):
        emb = sbert.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    np.save( config.save_file, embeddings)
    print(embeddings.shape)

train_set = DATAPROCESSING( train=True)
build_semantic(train_set)
