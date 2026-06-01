import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import CLIPModel

print("🚀 正在自动下载/加载 STL-10 数据集...")
# 标准预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 只要加了 download=True，它就会自动在 ./data 里下好！
train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

model = CLIPModel(model_name="ViT-B/32").cuda()
model.eval()

def extract_features(dataloader, desc):
    features, labels = [], []
    for x, y in dataloader:
        with torch.no_grad():
            features.append(model.encode_image(x.cuda()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

print("提取训练集特征...")
feat_train, lab_train = extract_features(train_loader, "Train")
print("提取测试集特征...")
feat_test, lab_test = extract_features(test_loader, "Test")

np.save("./data/stl10_image_embedding_train.npy", feat_train)
np.savetxt("./data/stl10_labels_train.txt", lab_train)
np.save("./data/stl10_image_embedding_test.npy", feat_test)
np.savetxt("./data/stl10_labels_test.txt", lab_test)

print(f"✅ STL-10 图像特征搞定！训练集: {feat_train.shape}, 测试集: {feat_test.shape}")