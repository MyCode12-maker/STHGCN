import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances

def evaluate_embedding(Z, num_clusters=5):
    # PCA降维
    pca = PCA(n_components=3)
    Z_3d = pca.fit_transform(Z)
    # KMeans 聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(Z_3d)
    labels = kmeans.labels_
    # 定量指标
    sil_score = silhouette_score(Z_3d, labels)
    intra_variances = [np.mean(np.linalg.norm(Z_3d[labels==k]-Z_3d[labels==k].mean(axis=0), axis=1)**2)
                       for k in range(num_clusters)]
    intra_var_mean = np.mean(intra_variances)
    centroids = np.array([Z_3d[labels==k].mean(axis=0) for k in range(num_clusters)])
    inter_cluster_mean = np.mean(pairwise_distances(centroids)[np.triu_indices(num_clusters, k=1)])
    return Z_3d, labels, sil_score, intra_var_mean, inter_cluster_mean

def plot_3d(Z_3d, labels, title):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(Z_3d[:,0], Z_3d[:,1], Z_3d[:,2], c=labels, cmap='tab10', s=50, edgecolor='k')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.colorbar(scatter, label='Cluster ID')
    plt.show()

# ===== 假设已有两个 embedding =====
Z_pois_fusion = torch.load("final_p_gate.pt").detach().cpu().numpy()
Z_pois_add = torch.load("final_p_gate_no.pt").detach().cpu().numpy()

# ===== 评估融合嵌入 =====
Z_3d_f, labels_f, sil_f, intra_f, inter_f = evaluate_embedding(Z_pois_fusion)
plot_3d(Z_3d_f, labels_f, "POI Embeddings: Frequency-Domain Gated Fusion with KAN")

# ===== 评估直接相加嵌入 =====
Z_3d_a, labels_a, sil_a, intra_a, inter_a = evaluate_embedding(Z_pois_add)
plot_3d(Z_3d_a, labels_a, "POI Embeddings: Simple Addition Fusion")

# ===== 输出指标对比 =====
print("Metric Comparison:")
print(f"{'Method':<30} {'Silhouette':<12} {'Intra-Var':<12} {'Inter-Dist':<12}")
print(f"{'Fusion (KAN)':<30} {sil_f:<12.4f} {intra_f:<12.4f} {inter_f:<12.4f}")
print(f"{'Simple Addition':<30} {sil_a:<12.4f} {intra_a:<12.4f} {inter_a:<12.4f}")
