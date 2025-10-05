import torch
import scipy.sparse as sp
from sklearn.cluster import KMeans
from collections import defaultdict
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
import numpy as np

def haversine_distance(lon1, lat1, lon2, lat2):
    """计算两点之间的 Haversine 距离（单位：km）"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球半径（单位：km）
    return c * r

def compute_poi_distance_matrix(csv_path, save_path=None):
    """
    读取POI CSV文件并计算两两POI之间的Haversine距离矩阵。

    Args:
        csv_path (str): CSV文件路径，必须包含 ['POI_id', 'longitude', 'latitude'] 三列。
        save_path (str, optional): 若指定，将结果保存为.npy文件。

    Returns:
        np.ndarray: POI距离矩阵，shape为 [num_pois, num_pois]。
    """
    # 1. 读取CSV文件
    df = pd.read_csv(csv_path)
    poi_ids = df['POI_id'].values
    longitudes = df['longitude'].values
    latitudes = df['latitude'].values

    num_pois = len(poi_ids)
    distance_matrix = np.zeros((num_pois, num_pois))

    # 2. 计算Haversine距离
    for i in tqdm(range(num_pois), desc="计算POI距离"):
        for j in range(num_pois):
            if i != j:
                distance = haversine_distance(longitudes[i], latitudes[i],
                                              longitudes[j], latitudes[j])
                distance_matrix[i, j] = distance
            else:
                distance_matrix[i, j] = 0.0  # 自己到自己距离为0

    # 3. 可选：保存为.npy文件
    if save_path is not None:
        np.save(save_path, distance_matrix)
        print(f"距离矩阵已保存至 {save_path}")

    return distance_matrix

def normalize_adj(adj):
    """
    归一化邻接矩阵 A 为 D^{-1/2} * A * D^{-1/2}

    :param adj: scipy.sparse csr_matrix, shape [num_pois, num_pois]
    :return: scipy.sparse.csr_matrix, 归一化拉普拉斯矩阵
    """
    adj = adj + sp.eye(adj.shape[0])  # 加单位矩阵 (self-loop)

    rowsum = np.array(adj.sum(1)).flatten()  # 节点度：每个POI的邻居数（含self-loop）
    d_inv_sqrt = np.power(rowsum, -0.5)  # D^{-1/2}
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 处理度为0

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 构建 D^{-1/2} 对角矩阵

    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt  # 返回 D^{-1/2} * A * D^{-1/2}

def count_user_hour_occurrences(user_day_list):
    """
    统计每个用户在24小时内每个小时出现的次数，以及所有用户全局总次数

    :param user_day_list: list，每个元素是一个用户的多天数据，每天是5个 numpy数组的列表
    :return:
        user_hour_counts: dict {user_id: {hour: count}}
        global_hour_counts: dict {hour: total_count}
    """
    user_hour_counts = {}  # 每个用户
    global_hour_counts = defaultdict(int)  # 全局统计

    for user_id, day_list in enumerate(user_day_list):
        hour_counts = defaultdict(int)  # 当前用户的24小时计数器

        for day in day_list:
            hour_seq = day[3]  # numpy array of hours for the day

            for hour in hour_seq:
                hour = int(hour)
                if 0 <= hour <= 23:  # 只统计合法小时
                    hour_counts[hour] += 1
                    global_hour_counts[hour] += 1  # 同时更新全局计数

        user_hour_counts[user_id] = dict(hour_counts)  # 转为普通dict避免defaultdict外泄

    return user_hour_counts, dict(global_hour_counts)
def get_time_slot(hour, time_slots):
    """
    根据聚类结果返回时间段编号
    :param hour: 当前小时（0~23）
    :param time_slots: dict, {cluster_id: [hour1, hour2, ...]}
    :return: int, cluster_id
    """
    for slot, hours in time_slots.items():
        if hour in hours:
            return slot
    return -1  # 错误处理

def generate_time_slots(global_hour_counts, n_clusters=4):
    """
    使用KMeans将24小时聚为n_clusters个时间段
    :param global_hour_counts: numpy数组 shape=[24]
    :return: dict, {cluster_id: [hour1, hour2, ...]}
    """
    counts = np.array([global_hour_counts[h] for h in range(24)]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(counts)
    labels = kmeans.labels_  # 每个小时的cluster id

    time_slots = {i: [] for i in range(n_clusters)}
    for hour, label in enumerate(labels):
        time_slots[label].append(hour)

    return time_slots


def build_user_poi_graph(user_day_list, test_set, num_pois):
    """
    构建用户-POI交互二分图 (UP Graph)

    Args:
        user_day_list: list, 每个元素是一个用户的 day_list（训练集）
        test_set: list, 测试集的用户轨迹
        num_pois: int，总POI数量

    Returns:
        A: scipy.sparse.csr_matrix, shape = [num_users + num_pois, num_users + num_pois]
           用户节点 [0, num_users)，POI节点 [num_users, num_users+num_pois)
        uid2idx: dict, 原始用户ID -> 连续用户索引
        idx2uid: dict, 连续用户索引 -> 原始用户ID
    """
    # --------------- 收集所有用户ID ----------------
    all_user_ids = set()
    for day_list in user_day_list + test_set:
        for day in day_list:
            user_seq = day[2]  # day[2] 是用户id序列
            all_user_ids.update(user_seq)

    unique_user_ids = sorted(all_user_ids)
    num_users = len(unique_user_ids)

    uid2idx = {uid: idx for idx, uid in enumerate(unique_user_ids)}
    idx2uid = {v: k for k, v in uid2idx.items()}

    # --------------- 构建用户-POI边 ----------------
    row, col, data = [], [], []

    for day_list in user_day_list:
        for day in day_list:
            poi_seq = day[0]
            user_seq = day[2]
            for uid, poi in zip(user_seq, poi_seq):
                u_idx = uid2idx[int(uid)]
                p_idx = num_users + int(poi)  # POI节点偏移
                # 用户 -> POI
                row.append(u_idx)
                col.append(p_idx)
                data.append(1)
                # POI -> 用户（无向图对称）
                row.append(p_idx)
                col.append(u_idx)
                data.append(1)

    # --------------- 构建邻接矩阵 ----------------
    A = sp.csr_matrix((data, (row, col)), shape=(num_users + num_pois, num_users + num_pois))
    A_norm = normalize_adj(A)
    A_norm_torch = transform_csr_matrix_to_tensor(A_norm)

    return A_norm_torch, uid2idx

def build_global_user_poi_time_distance_rest_hypergraph_list(user_day_list,test_set, num_pois, distance_matrix, distance_threshold):
    """
    支持输入为 list 版本的 user_day_list
    构建全局 用户-POI-时间-空间-休息日+每个用户超边 超图
    超边数量 = 6时间 + 1空间 + 2休息日 + N用户 = 9 + num_users

    Args:
        user_day_list: list，每个元素是一个用户的 day_list
        num_pois: int，总POI数量
        distance_matrix: np.ndarray, [num_pois, num_pois]
        distance_threshold: float

    Returns:
        H: scipy.sparse.csr_matrix, shape = [num_pois, 9 + num_users]
        uid2col: dict, 原始用户ID -> 超图列索引
        col2uid: dict, 超图列索引 -> 原始用户ID
    """
    # ---------------- 收集所有用户ID ----------------
    all_user_ids = set()
    for day_list in user_day_list:
        for day in day_list:
            user_seq = day[2]  # day[2] 是用户ID序列
            all_user_ids.update(user_seq)
    for day_list in test_set:
        for day in day_list:
            user_seq = day[2]  # day[2] 是用户ID序列
            all_user_ids.update(user_seq)

    unique_user_ids = sorted(all_user_ids)
    num_users = len(unique_user_ids)

    # 用户ID到超图列索引映射
    user_edge_offset = 9
    uid2col = {uid: user_edge_offset + idx for idx, uid in enumerate(unique_user_ids)}
    col2uid = {v: k for k, v in uid2col.items()}

    # ---------------- 统计时间 ----------------
    user_hour_counts, global_hour_counts = count_user_hour_occurrences(user_day_list)
    time_slots = generate_time_slots(global_hour_counts, n_clusters=6)

    time_slot_edges = {slot: set() for slot in range(6)}
    rest_day_pois, non_rest_day_pois = set(), set()
    user_pois = {uid: set() for uid in unique_user_ids}

    # ---------------- 遍历轨迹 ----------------
    for day_list in user_day_list:
        for day in day_list:
            poi_seq = day[0]
            hour_seq = day[3]
            rest_flag_seq = day[4]
            user_seq = day[2]

            for i in range(len(poi_seq)):
                poi = int(poi_seq[i])
                hour = int(hour_seq[i])
                slot = get_time_slot(hour, time_slots)
                time_slot_edges[slot].add(poi)

                if int(rest_flag_seq[i]) == 1:
                    rest_day_pois.add(poi)
                else:
                    non_rest_day_pois.add(poi)

                uid = int(user_seq[i])
                user_pois[uid].add(poi)

    # ---------------- 空间超边 ----------------
    all_pois = set().union(*time_slot_edges.values(), rest_day_pois, non_rest_day_pois)
    all_pois = list(all_pois)
    spatial_edge = set()
    for i in range(len(all_pois)):
        for j in range(i + 1, len(all_pois)):
            poi_i, poi_j = all_pois[i], all_pois[j]
            if distance_matrix[poi_i, poi_j] <= distance_threshold:
                spatial_edge.add(poi_i)
                spatial_edge.add(poi_j)

    # ---------------- 构建H矩阵 ----------------
    row, col, data = [], [], []

    # 前9条公共超边
    hyperedges = []
    for slot in range(6):
        hyperedges.append(list(time_slot_edges[slot]))
    hyperedges.append(list(spatial_edge))
    hyperedges.append(list(rest_day_pois))
    hyperedges.append(list(non_rest_day_pois))

    for edge_id, poi_list in enumerate(hyperedges):
        for poi in poi_list:
            row.append(poi)
            col.append(edge_id)
            data.append(1)

    # 用户超边
    for uid, poi_set in user_pois.items():
        col_idx = uid2col[uid]
        for poi in poi_set:
            row.append(poi)
            col.append(col_idx)
            data.append(1)

    H = sp.csr_matrix((data, (row, col)), shape=(num_pois, user_edge_offset + num_users))

    return H, uid2col

def get_hyper_deg(incidence_matrix):
    """
    计算超图节点度的倒数对角矩阵 D^{-1}

    Args:
        incidence_matrix: scipy.sparse 矩阵, shape [num_nodes, num_hyperedges]

    Returns:
        d_mat_inv: scipy.sparse 对角矩阵 D^{-1}
    """
    rowsum = np.array(incidence_matrix.sum(1)).flatten().astype(float)  # 转为 float
    rowsum[rowsum == 0] = 1.0  # 防止除0
    d_inv = np.power(rowsum, -1)  # 计算倒数
    d_mat_inv = sp.diags(d_inv)  # 构成对角矩阵

    return d_mat_inv
def transform_csr_matrix_to_tensor(sparse_mx):
    """
    scipy sparse csr_matrix → PyTorch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def get_hypergraph_laplacian(H):
    """
    计算超图归一化拉普拉斯矩阵 L = I - Dv^{-1/2} * H * De^{-1} * H^T * Dv^{-1/2}

    :param H: scipy csr_matrix, shape [num_pois, num_hyperedges]
    :return: scipy csr_matrix, 拉普拉斯矩阵 [num_pois, num_pois]
    """
    num_pois, num_edges = H.shape

    # 顶点度 Dv: shape [num_pois, num_pois]
    Dv = np.array(H.sum(axis=1)).flatten()
    Dv_inv_sqrt = np.power(Dv, -0.5)
    Dv_inv_sqrt[np.isinf(Dv_inv_sqrt)] = 0.0  # 处理除零
    Dv_inv_sqrt_mat = sp.diags(Dv_inv_sqrt)

    # 超边度 De: shape [num_edges, num_edges]
    De = np.array(H.sum(axis=0)).flatten()
    De_inv = np.power(De, -1.0)
    De_inv[np.isinf(De_inv)] = 0.0
    De_inv_mat = sp.diags(De_inv)

    # 拉普拉斯矩阵 L = I - Dv^{-1/2} * H * De^{-1} * H^T * Dv^{-1/2}
    I = sp.identity(num_pois)
    L = I - Dv_inv_sqrt_mat @ H @ De_inv_mat @ H.T @ Dv_inv_sqrt_mat  # [num_pois, num_pois]

    return L
def build_local_user_poi_hourly_spatial_rest_hypergraphs_with_norm(user_day_list, num_pois, distance_matrix, distance_threshold):
    """
    为每个用户构建归一化后的局部 用户-POI-时间-空间-休息日 超图（每个用户27个超边）

    :param user_day_list: list，每个元素为一个用户的多天数据（每天5个 numpy数组）
    :param num_pois: int, POI总数
    :param distance_matrix: numpy.ndarray, shape [num_pois, num_pois], 预计算的POI距离矩阵（单位km）
    :param distance_threshold: float, 距离阈值，控制哪些POI归为同一空间超边
    :return: user_hypergraphs: dict，键为user_id，值为归一化的超图 D^{-1}H（shape: [num_pois, 27]）
    """
    user_hypergraphs = {}

    for user_id, day_list in enumerate(user_day_list):
        user_hour_edges = {h: set() for h in range(24)}  # 24小时超边
        user_all_pois = set()  # 用户访问过的所有POI
        rest_day_pois = set()  # 休息日访问的POI
        non_rest_day_pois = set()  # 非休息日访问的POI

        for day in day_list:
            poi_seq = day[0]  # POI序列
            hour_seq = day[3]  # 小时序列
            rest_flag_seq = day[4]  # 休息日标记序列（0/1）

            for i in range(len(poi_seq)):
                poi = int(poi_seq[i])
                hour = int(hour_seq[i]) % 24
                rest_flag = int(rest_flag_seq[i])

                user_hour_edges[hour].add(poi)
                user_all_pois.add(poi)

                if rest_flag == 1:
                    rest_day_pois.add(poi)
                else:
                    non_rest_day_pois.add(poi)

        # ====== 构建空间超边 ======
        spatial_edge = set()
        all_pois = list(user_all_pois)
        for i in range(len(all_pois)):
            for j in range(i + 1, len(all_pois)):
                poi_i = all_pois[i]
                poi_j = all_pois[j]
                if distance_matrix[poi_i, poi_j] <= distance_threshold:
                    spatial_edge.add(poi_i)
                    spatial_edge.add(poi_j)

        # ====== 构建H矩阵 (shape: [num_pois, 27]) ======
        row, col, data = [], [], []

        # 24小时超边
        for hour in range(24):
            poi_list = list(user_hour_edges[hour])
            for poi in poi_list:
                row.append(poi)
                col.append(hour)  # hour: 0 ~ 23
                data.append(1)

        # 空间超边（第24列）
        for poi in list(spatial_edge):
            row.append(poi)
            col.append(24)
            data.append(1)

        # 休息日超边（第25列）
        for poi in list(rest_day_pois):
            row.append(poi)
            col.append(25)
            data.append(1)

        # 非休息日超边（第26列）
        for poi in list(non_rest_day_pois):
            row.append(poi)
            col.append(26)
            data.append(1)

        # 构建稀疏矩阵
        H_user = sp.csr_matrix((data, (row, col)), shape=(num_pois, 27))

        # ====== 归一化 D^{-1}H ======
        H_norm = get_hypergraph_laplacian(H_user)
        H_norm = transform_csr_matrix_to_tensor(H_norm)

        user_hypergraphs[user_id] = H_norm  # 存储每个用户的归一化局部超图

    return user_hypergraphs

def build_local_user_poi_time_distance_rest_hypergraphs_with_norm(user_day_list, num_pois, distance_matrix, distance_threshold):
    """
    为每个用户构建归一化后的局部 用户-POI-时间-空间-休息日 超图（每个用户4个时间段超边 + 1个空间超边 + 2个休息日超边）

    Args:
        user_day_list: list，每个元素是一个用户的多天数据，每天是5个numpy数组的列表
        num_pois: int，POI总数
        distance_matrix: numpy.ndarray，shape [num_pois, num_pois]，预计算的POI距离矩阵（单位km）
        distance_threshold: float，距离阈值（单位km）

    Returns:
        user_hypergraphs: dict，键为user_id，值为归一化后的超图D^{-1}H (shape: [num_pois, 7])
    """
    user_hour_counts, global_hour_counts = count_user_hour_occurrences(user_day_list)

    # 2. 聚类获得动态时间段
    time_slots = generate_time_slots(global_hour_counts, n_clusters=6)
    user_hypergraphs = {}

    for user_id, day_list in enumerate(user_day_list):
        user_time_slot_edges = {0: set(), 1: set(), 2: set(), 3: set(),4: set(), 5: set()}
        rest_day_pois = set()
        non_rest_day_pois = set()

        # 统计时间段POI和休息日POI
        for day in day_list:
            poi_seq = day[0]
            hour_seq = day[3]
            rest_flag_seq = day[4]  # 1为休息日，0为非休息日

            for i in range(len(poi_seq)):
                poi = int(poi_seq[i])
                hour = int(hour_seq[i])
                slot = get_time_slot(hour,time_slots)
                user_time_slot_edges[slot].add(poi)

                rest_flag = int(rest_flag_seq[i])
                if rest_flag == 1:
                    rest_day_pois.add(poi)
                else:
                    non_rest_day_pois.add(poi)

        # 构建空间超边：用户访问过的POI里，距离小于阈值的POI连成空间超边
        all_pois_visited = set()
        for slot in range(4):
            all_pois_visited.update(user_time_slot_edges[slot])
        all_pois_visited.update(rest_day_pois)
        all_pois_visited.update(non_rest_day_pois)
        all_pois_visited = list(all_pois_visited)

        spatial_edge = set()
        for i in range(len(all_pois_visited)):
            for j in range(i+1, len(all_pois_visited)):
                poi_i = all_pois_visited[i]
                poi_j = all_pois_visited[j]
                if distance_matrix[poi_i, poi_j] <= distance_threshold:
                    spatial_edge.add(poi_i)
                    spatial_edge.add(poi_j)

        # 构建超图H，7个超边
        row, col, data = [], [], []

        for slot in range(4):
            for poi in user_time_slot_edges[slot]:
                row.append(poi)
                col.append(slot)
                data.append(1)

        # 空间超边编号为4
        for poi in spatial_edge:
            row.append(poi)
            col.append(4)
            data.append(1)

        # 休息日超边编号为5
        for poi in rest_day_pois:
            row.append(poi)
            col.append(5)
            data.append(1)

        # 非休息日超边编号为6
        for poi in non_rest_day_pois:
            row.append(poi)
            col.append(6)
            data.append(1)

        H_user = sp.csr_matrix((data, (row, col)), shape=(num_pois, 7))

        # 归一化 D^{-1}H
        H_norm = get_hypergraph_laplacian(H_user)
        H_norm = transform_csr_matrix_to_tensor(H_norm)

        user_hypergraphs[user_id] = H_norm

    return user_hypergraphs