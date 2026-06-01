import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import faiss
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
import torch

BICUBIC = InterpolationMode.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transforms(dataset="CIFAR-10"):
    if dataset in ["CIFAR-10", "CIFAR-20", "STL-10", "DTD", "UCF101"]:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=BICUBIC),
            torchvision.transforms.CenterCrop(224),
            _convert_image_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interpolation=BICUBIC),
            torchvision.transforms.CenterCrop(224),
            _convert_image_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    return transforms

def get_dataloader(dataset="CIFAR-10", batch_size=4096):
    transforms = get_transforms(dataset)
    if dataset == "CIFAR-10":
        data_train = CIFAR10(root="./data", train=True, download=True, transform=transforms)
        data_test = CIFAR10(root="./data", train=False, download=True, transform=transforms)
    elif dataset == "CIFAR-20":
        data_train = CIFAR100(root="./data", train=True, download=True, transform=transforms)
        data_test = CIFAR100(root="./data", train=False, download=True, transform=transforms)
    else:
        raise NotImplementedError

    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=False, drop_last=False)
    dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False, drop_last=False)
    return dataloader_train, dataloader_test

def mine_nearest_neighbors(features, topk=50, index_file=None):
    print("Computing nearest neighbors...")
    features = features.astype(np.float32)
    faiss.normalize_L2(features) 
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  
    print("Nearest neighbors computed.")
    return indices[:, 1:]

class NeighborsDataset(Dataset):
    def __init__(self, dataset_text, dataset_image, indices_text, indices_image, k):
        super(NeighborsDataset, self).__init__()
        self.num_neighbors = k
        self.dataset_text = dataset_text
        self.dataset_image = dataset_image
        self.indices_text = indices_text
        self.indices_image = indices_image

    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        anchor_text = self.dataset_text.__getitem__(index)  
        anchor_image = self.dataset_image.__getitem__(index)  
        neighbor_index_text = np.random.choice(self.indices_text[index], 1)[0]  
        neighbor_index_image2text = np.random.choice(self.indices_image[index], 1)[0]  
        neighbor_text = self.dataset_text.__getitem__(neighbor_index_text)  
        neighbor_image2text = self.dataset_text.__getitem__(neighbor_index_image2text)
        neighbor_index_image = np.random.choice(self.indices_image[index], 1)[0]  
        neighbor_index_text2image = np.random.choice(self.indices_text[index], 1)[0]  
        neighbor_text2image = self.dataset_image.__getitem__(neighbor_index_text2image)
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image)  
        return index, anchor_text, anchor_image, neighbor_text, neighbor_image, neighbor_image2text, neighbor_text2image

class TestDataset(Dataset):
    def __init__(self, dataset_text, dataset_image):
        super(TestDataset, self).__init__()
        self.dataset_text = dataset_text
        self.dataset_image = dataset_image
        
    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        return self.dataset_text.__getitem__(index), self.dataset_image.__getitem__(index)