import clip
import torch
from torch import nn
from timm.layers import DropPath, trunc_normal_
import torch.nn.functional as F

class ClusterHead(nn.Module):
    def __init__(self, in_dim=512,text_in_dim=768, num_clusters=10,num_losses=3,proto_momentum=0.9,device="cuda"):
        super().__init__()
        self.num_clusters = num_clusters
        self.text_proj = nn.Linear(text_in_dim, in_dim)
        #for text cluster probability computation
        self.cluster_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
            
        )
        #for image cluster probability computation
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
    def get_weights(self):
        return self.weight_generator()
    def forward(self, text, image):
        if text.shape[1]!=512:
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
    # 新加
    def compute_consistency_matrix(self, logit_text, logit_image):
        # Normalize the logits to have unit norm
        logit_text = F.normalize(logit_text, p=2, dim=1)
        logit_image = F.normalize(logit_image, p=2, dim=1)
        
        # Compute the cosine similarity matrix
        consistency_matrix = torch.mm(logit_text, logit_image.t())
        return consistency_matrix


