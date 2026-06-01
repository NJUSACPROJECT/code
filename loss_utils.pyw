import torch
from torch import nn
import torch.nn.functional as F

def entropy(logit):
    logit = logit.mean(dim=0)
    logit_ = torch.clamp(logit, min=1e-9)
    b = logit_ * torch.log(logit_)
    return -b.sum()

def consistency_loss(anchors, neighbors):
    b, n = anchors.size()
    similarity = torch.bmm(anchors.view(b, 1, n), neighbors.view(b, n, 1)).squeeze()
    ones = torch.ones_like(similarity)
    consistency_loss = F.binary_cross_entropy(similarity, ones)
    return consistency_loss

def num_consistency_loss(anchors, neighbors):
    anchors = anchors.t()  
    neighbors = neighbors.t()
    similarity = torch.sum(anchors * neighbors, dim=1) 
    loss = F.binary_cross_entropy_with_logits(similarity, torch.ones_like(similarity))
    return loss

class DataContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, max_epoch=20, alpha=1.0, eps=1e-8):
        super().__init__()
        self.base_temp = temperature
        self.ratio = nn.Parameter(torch.tensor(alpha))  
        self.max_epoch = max_epoch
        self.alpha = alpha
        self.eps = eps

    def compute_base(self, x, y, curr_temp, batch_size):
        features = torch.cat([x, y], dim=0) 
        sim_matrix = torch.mm(features, features.T)  
        sim_matrix = sim_matrix / curr_temp
        pos_sim = torch.diag(sim_matrix, batch_size)  
        pos_sim = torch.cat([pos_sim, pos_sim])       
        w_base = torch.sigmoid(self.alpha * pos_sim)  
        return w_base
    
    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0) 
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        return mask.bool()

    def forward(self, c_i, c_j, c_x, c_y, c_a, c_b, epoch, ifweights, ratio):
        batch_size = c_i.size(0)
        curr_temp = self.base_temp 
        c_i = F.normalize(c_i, p=1, dim=1)  
        c_j = F.normalize(c_j, p=1, dim=1)
        c_x = F.normalize(c_x, p=1, dim=1)
        c_y = F.normalize(c_y, p=1, dim=1)
        c_a = F.normalize(c_a, p=1, dim=1)
        c_b = F.normalize(c_b, p=1, dim=1)

        w_ratio = ratio 
        w_base1 = self.compute_base(c_a, c_b, curr_temp, batch_size)
        w_base2 = self.compute_base(c_x, c_y, curr_temp, batch_size)
        w_base = w_ratio * w_base1 + (1 - w_ratio) * w_base2

        features = torch.cat([c_i, c_j], dim=0) 
        sim_matrix = torch.mm(features, features.T) 
        sim_matrix = sim_matrix / curr_temp

        labels = torch.arange(batch_size, device=c_i.device)
        labels = torch.cat([labels + batch_size, labels])  
        
        logits = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        loss = -logits[torch.arange(2 * batch_size), labels]  

        if ifweights:
            combined_weights = w_base
            weighted_loss = (combined_weights * loss).sum() / (combined_weights.sum() + 1e-8)
        else:
            weighted_loss = loss.mean()
        
        return weighted_loss