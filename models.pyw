import clip
import torch
from torch import nn
from timm.layers import trunc_normal_
import torch.nn.functional as F

class ClusterHead(nn.Module):
    def __init__(self, in_dim=512, text_in_dim=768, num_clusters=10, num_losses=3, proto_momentum=0.9, device="cuda"):
        super().__init__()
        self.num_clusters = num_clusters
        self.text_proj = nn.Linear(text_in_dim, in_dim)
        self.cluster_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        self.cluster_head_image = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        trunc_normal_(self.cluster_head_text[0].weight, std=0.02)
        trunc_normal_(self.cluster_head_text[3].weight, std=0.02)
        trunc_normal_(self.cluster_head_image[0].weight, std=0.02)
        trunc_normal_(self.cluster_head_image[3].weight, std=0.02)

    def forward(self, text, image):
        if text.shape[1] != 512:
            text = self.text_proj(text)
        logit_text = self.cluster_head_text(text)
        logit_image = self.cluster_head_image(image)
        return logit_text, logit_image

    def forward_embedding(self, image):
        embedding = self.cluster_head_image[0](image)
        embedding = self.cluster_head_image[1](embedding)
        embedding = self.cluster_head_image[2](embedding)
        embedding = self.cluster_head_image[3](embedding)
        return embedding

    def compute_consistency_matrix(self, logit_text, logit_image):
        logit_text = F.normalize(logit_text, p=2, dim=1)
        logit_image = F.normalize(logit_image, p=2, dim=1)
        consistency_matrix = torch.mm(logit_text, logit_image.t())
        return consistency_matrix

class CLIPModel(nn.Module):
    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device="cpu")

    def encode_image(self, image):
        return self.model.encode_image(image).float()