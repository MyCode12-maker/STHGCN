"""
2D embedding visualization utilities.

函数:
- plot_2d_embedding_matplotlib(emb, labels=None, categories=None, ...)
- plot_2d_embedding_interactive(emb, labels=None, categories=None, ...)

输入:
- emb: numpy.ndarray, shape (N,2)
- labels: (optional) array-like length N, 文本标签（用于鼠标悬停或标注）
- categories: (optional) array-like length N，类别（用于上色）。若给了 categories，会按类别着色并显示图例。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

# interactive
import plotly.express as px

# clustering optional
from sklearn.cluster import KMeans

def _ensure_2d_emb(emb):
    emb = np.asarray(emb)
    if emb.ndim != 2 or emb.shape[1] != 2:
        raise ValueError("emb must be shape (N, 2). Got: {}".format(emb.shape))
    return emb

def plot_2d_embedding_matplotlib(
    emb,
    categories=None,
    labels=None,
    figsize=(8,6),
    s=30,
    alpha=0.8,
    annotate=False,
    show_convex_hull=False,
    add_kmeans=None,
    title="2D embedding (Matplotlib)",
    savepath=None
):
    """
    静态散点图（matplotlib）。
    - categories: 可选，按类别着色（长度 N）
    - labels: 可选，用于 annotate=True 时对点做文字标注
    - annotate: 若 True，显示每个点的 labels（谨慎用于点较少的情况）
    - show_convex_hull: 若 True，对每个类别绘制 convex hull（需至少3个点）
    - add_kmeans: 若为 int k，会在图上绘制 k-means 类别中心（和按类着色互斥推荐）
    """
    emb = _ensure_2d_emb(emb)
    N = emb.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    if categories is None and add_kmeans is None:
        ax.scatter(emb[:,0], emb[:,1], s=s, alpha=alpha)
    else:
        if add_kmeans is not None:
            k = int(add_kmeans)
            km = KMeans(n_clusters=k, random_state=42).fit(emb)
            cat = km.labels_
            centers = km.cluster_centers_
            unique = np.unique(cat)
            cmap = cm.get_cmap('tab10', len(unique))
            for i,u in enumerate(unique):
                idx = (cat==u)
                ax.scatter(emb[idx,0], emb[idx,1], s=s, alpha=alpha, label=f'cluster {u}', cmap=None)
            ax.scatter(centers[:,0], centers[:,1], s=150, marker='X', edgecolor='k', linewidth=1.2)
            ax.legend()
        else:
            cats = np.asarray(categories)
            unique = np.unique(cats)
            cmap = cm.get_cmap('tab10', len(unique))
            legend_handles = []
            for i,u in enumerate(unique):
                idx = (cats == u)
                ax.scatter(emb[idx,0], emb[idx,1], s=s, alpha=alpha, label=str(u))
                if show_convex_hull and idx.sum() >= 3:
                    try:
                        hull = ConvexHull(emb[idx])
                        hull_pts = emb[idx][hull.vertices]
                        ax.plot(np.append(hull_pts[:,0], hull_pts[0,0]),
                                np.append(hull_pts[:,1], hull_pts[0,1]),
                                linestyle='--', linewidth=1)
                    except Exception:
                        pass
                legend_handles.append(Patch(facecolor=cmap(i), label=str(u)))
            ax.legend()
    if annotate and labels is not None:
        for i, txt in enumerate(labels):
            ax.annotate(str(txt), (emb[i,0], emb[i,1]), fontsize=8, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(False)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()


def plot_2d_embedding_interactive(
    emb,
    categories=None,
    labels=None,
    hover_template=None,
    title="2D embedding (interactive)",
    width=800,
    height=600
):
    """
    交互式绘图（Plotly）— 支持悬停、缩放和导出为 HTML。
    - hover_template: 自定义悬停信息模板，例如 "index: %{customdata[0]}<br>label: %{customdata[1]}"
    - labels 会作为 customdata 传递到 hover（若 labels 为 None，则传入索引）
    - categories 会作为 color 用于着色
    """
    emb = _ensure_2d_emb(emb)
    N = emb.shape[0]
    xs = emb[:,0]
    ys = emb[:,1]
    if labels is None:
        labels_plot = [str(i) for i in range(N)]
    else:
        labels_plot = [str(x) for x in labels]
    df = {
        "x": xs,
        "y": ys,
        "label": labels_plot
    }
    if categories is not None:
        df["category"] = categories

    if categories is not None:
        fig = px.scatter(df, x="x", y="y", color="category", hover_data=["label"], title=title, width=width, height=height)
    else:
        fig = px.scatter(df, x="x", y="y", hover_data=["label"], title=title, width=width, height=height)

    if hover_template is not None:
        fig.update_traces(hovertemplate=hover_template)
    fig.update_layout(legend_title_text='category' if categories is not None else None)
    fig.show()
    return fig
import numpy as np
# 生成 demo 数据
N = 200
np.random.seed(0)
emb = np.vstack([np.random.randn(N//2,2) + np.array([2,0]), np.random.randn(N//2,2) + np.array([-2,0])])
cats = np.array(['A']*(N//2) + ['B']*(N//2))
labels = [f'pt{i}' for i in range(N)]

# matplotlib 静态图（含 convex hull）
plot_2d_embedding_matplotlib(emb, categories=cats, labels=labels, annotate=False, show_convex_hull=True, title="示例：2D embedding")

# plotly 交互式
fig = plot_2d_embedding_interactive(emb, categories=cats, labels=labels)
# 若想把交互式图保存为 html:
# fig.write_html("embedding_plot.html")
