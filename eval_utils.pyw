import numpy as np
from sklearn import metrics
from munkres import Munkres

def cluster_metric(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    
    # 动态获取安全的最大聚类数量，强制 >= 20 保证安全
    n_clusters = max(int(np.max(label)) + 1, int(np.max(pred)) + 1, 20)
    
    pred_adjusted = get_y_preds(label, pred, n_clusters)
    acc = metrics.accuracy_score(pred_adjusted, label)
    print(
        "[Clustering Result]: ACC = {:.2f}, NMI = {:.2f}, ARI = {:.2f}".format(
            acc * 100, nmi * 100, ari * 100
        )
    )
    return acc, nmi, ari

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])  
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_cluster_labels_from_indices(indices, n_clusters):
    cluster_labels = np.zeros(n_clusters)
    for i in range(len(indices)):
        row, col = indices[i]
        cluster_labels[row] = col
    return cluster_labels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    confusion_matrix = metrics.confusion_matrix(
        y_true, cluster_assignments, labels=list(range(n_clusters))
    )
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices, n_clusters)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred