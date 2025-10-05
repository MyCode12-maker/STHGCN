import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ===== 假设已有嵌入 =====
M, N, d = 5000, 20, 32
Z_users = torch.randn(M, d)
Z_pois = torch.load("final_p_gate_no.pt")
Z_pois = Z_pois.detach().cpu().numpy()
Z_pois = Z_pois[:, :32]
# ===== 抽样函数 =====
def sample_embeddings(Z_users, Z_pois, num_users=100, num_pois=20):
    # 直接选前 num_users 和前 num_pois
    user_idx = torch.arange(min(num_users, Z_users.shape[0]))
    poi_idx = torch.arange(min(num_pois, Z_pois.shape[0]))
    return Z_users[user_idx], Z_pois[poi_idx]


# ===== 抽样 =====
Z_users_sample, Z_pois_sample = sample_embeddings(Z_users, Z_pois, num_users=20, num_pois=20)

# ===== 计算相似度矩阵 =====
#sim_matrix = cosine_similarity(Z_users_sample.numpy(), Z_pois_sample.numpy())  # [50, 60]


# ===== 绘制小方格热力图 =====
plt.figure(figsize=(12, 8))  # figure 可以和 grid 比例一致
sns.heatmap(
    Z_pois_sample,
    cmap="coolwarm",
    cbar=True,
    square=True,          # 每个格子为正方形
    linewidths=0.1,
    linecolor="white",
    xticklabels=np.arange(d),
    yticklabels=np.arange(N)
)
plt.title("POI Embedding Heatmap (Sampled 20 POIs)", fontsize=14)
plt.xlabel("Embedding Dimension", fontsize=12)
plt.ylabel("POI Index", fontsize=12)
plt.tight_layout()
plt.show()
