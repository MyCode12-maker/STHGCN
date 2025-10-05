import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载高维嵌入
X = torch.load("hg_pois_embs_no.pt")

# 转为 numpy
X_np = X.detach().cpu().numpy()

# t-SNE 降维到 2D，参数调整以增强聚簇效果
tsne = TSNE(
    n_components=2,
    random_state=10,
    perplexity=5,      # 较小的 perplexity 强化局部聚簇
    learning_rate=300, # 可以尝试 100~500
    n_iter=2000,       # 增加迭代次数，使聚簇更稳定
    init='pca',        # PCA 初始化通常更稳定
)
X_2d = tsne.fit_transform(X_np)

# 绘制散点图
plt.figure(figsize=(6,6))
plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    s=40,
    alpha=0.8,
    color="royalblue",   # 换颜色，这里用深红色
    edgecolors='k'
)


plt.grid(True, linestyle="--", alpha=0.5)

# 保存为 PNG 文件，dpi=300
plt.savefig("tsne_embedding_no.png", dpi=300, bbox_inches="tight")

plt.show()
