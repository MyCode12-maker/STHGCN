import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances


# ===== 嵌入评估函数 =====
def evaluate_embedding_2d(Z, num_clusters=10):
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(Z_2d)
    labels = kmeans.labels_

    sil_score = silhouette_score(Z_2d, labels)

    # 簇内平均方差
    intra_variances = [np.mean(np.linalg.norm(Z_2d[labels == k] - Z_2d[labels == k].mean(axis=0), axis=1) ** 2)
                       for k in range(num_clusters)]
    intra_var_mean = np.mean(intra_variances)

    # 簇间平均距离
    centroids = np.array([Z_2d[labels == k].mean(axis=0) for k in range(num_clusters)])
    inter_cluster_mean = np.mean(pairwise_distances(centroids)[np.triu_indices(num_clusters, k=1)])

    return Z_2d, labels, sil_score, intra_var_mean, inter_cluster_mean


# ===== 绘制 2D PCA 散点图 =====
def plot_2d(Z_2d, labels, save_path=None):
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1],
                          c=labels, cmap='tab10',
                          s=50, edgecolor='k')
    plt.grid(True)
    plt.colorbar(scatter, label='Cluster ID')
    plt.axis('equal')  # 保持比例
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存为png，DPI 300
    plt.show()


# ===== 假设已有两个 embedding =====
Z_pois_fusion = torch.load("final_p_gate.pt").detach().cpu().numpy()
Z_pois_add = torch.load("final_p_gate_no.pt").detach().cpu().numpy()

# ===== 评估融合嵌入 =====
Z_2d_f, labels_f, sil_f, intra_f, inter_f = evaluate_embedding_2d(Z_pois_fusion)
plot_2d(Z_2d_f, labels_f, save_path="fusion_poi.png")
# ===== 评估直接相加嵌入 =====
Z_2d_a, labels_a, sil_a, intra_a, inter_a = evaluate_embedding_2d(Z_pois_add)
plot_2d(Z_2d_a, labels_a, save_path="simple_addition_poi.png")

# ===== 输出指标对比 =====
print("Metric Comparison:")
print(f"{'Method':<30} {'Silhouette':<12} {'Intra-Var':<12} {'Inter-Dist':<12}")
print(f"{'Fusion (KAN)':<30} {sil_f:<12.4f} {intra_f:<12.4f} {inter_f:<12.4f}")
print(f"{'Simple Addition':<30} {sil_a:<12.4f} {intra_a:<12.4f} {inter_a:<12.4f}")
