import torch
from torch import nn

import settings

device = settings.gpuId if torch.cuda.is_available() else 'cpu'
from Graph_conv import *
from ChebyKANLayer import *
class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features,order):
        super().__init__()
        self.fc1 = ChebyKANLinear(
                            in_features,
                            out_features,
                            order)
    def forward(self, x):
        #x = torch.unsqueeze(x, 0)
        x = self.fc1(x)
        return x
class GatedKAN1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, KANLayer):
        super(GatedKAN1, self).__init__()
        # 用KAN替代Linear
        self.gate_net_real = KANLayer(input_dim, hidden_dim, 1)  # 实部门控
        self.gate_net_imag = KANLayer(input_dim, hidden_dim, 1)  # 虚部门控
    def forward(self, x1, x2):
        # FFT
        X1_f = torch.fft.fft(x1, dim=-1)
        X2_f = torch.fft.fft(x2, dim=-1)

        # 分离实部和虚部
        X1_real, X1_imag = X1_f.real, X1_f.imag
        X2_real, X2_imag = X2_f.real, X2_f.imag

        # 实部门控融合
        gate_real = torch.sigmoid(self.gate_net_real(X1_real) + self.gate_net_real(X2_real))
        fused_real = gate_real * X1_real + (1 - gate_real) * X2_real

        # 虚部门控融合
        gate_imag = torch.sigmoid(self.gate_net_imag(X1_imag) + self.gate_net_imag(X2_imag))
        fused_imag = gate_imag * X1_imag + (1 - gate_imag) * X2_imag

        # 重构复数
        X_fused = torch.complex(fused_real, fused_imag)

        # IFFT 回时域
        x_out = torch.fft.ifft(X_fused, dim=-1).real

        return x_out

class STHGCN(nn.Module):
    def __init__(
            self,
            vocab_size,
            f_embed_size,
            layers,
    ):
        super().__init__()

        self.loss_func = nn.CrossEntropyLoss()

        self.gate1 = GatedKAN1(f_embed_size, f_embed_size, f_embed_size,
                             lambda in_dim, hid_dim, out_dim: ChebyKANLayer(f_embed_size, f_embed_size, order=3))

        self.poi_embedding = nn.Embedding(vocab_size["POI"] + 1, f_embed_size, padding_idx=vocab_size["POI"])
        self.UI_embedding = nn.Embedding(vocab_size["POI"] + 1 + 42, f_embed_size, padding_idx=vocab_size["POI"]+42) #42  764 786
        nn.init.xavier_uniform_(self.UI_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        self.mv_hconv_network = MultiViewHyperConvNetwork(layers, f_embed_size, 0.1, device)
        self.pp_ConvNetwork = GeoConvNetwork(layers, 0.1)

        self.w_gate_col = nn.Parameter(torch.FloatTensor(f_embed_size,f_embed_size))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, f_embed_size))
        self.w_gate_U = nn.Parameter(torch.FloatTensor(f_embed_size, f_embed_size))
        self.b_gate_U = nn.Parameter(torch.FloatTensor(1, f_embed_size))
        nn.init.xavier_normal_(self.w_gate_U.data)
        nn.init.xavier_normal_(self.b_gate_U.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        self.a = nn.Parameter(torch.tensor(0.5))
    def graph_contrastive_loss(self,emb1, emb2, temperature=0.2):
        """
        对比学习损失（节点级）：
        emb1, emb2: [N, D] 两个图的节点嵌入
        temperature: 温度参数
        """
        # 1. 归一化
        z1 = F.normalize(emb1, dim=-1)
        z2 = F.normalize(emb2, dim=-1)

        # 2. 相似度矩阵 [N, N]
        sim_matrix = torch.mm(z1, z2.T)  # cosine similarity

        # 3. 构造正样本对
        N = z1.size(0)
        labels = torch.arange(N, device=emb1.device)

        # 4. 对比学习 loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = (loss_fn(sim_matrix / temperature, labels) +
                loss_fn(sim_matrix.T / temperature, labels)) / 2

        return loss

    def graph_level_contrastive_loss(self,graph_emb1, graph_emb2, temperature=0.2):
        """
        图级别对比学习损失（Graph-level Contrastive Learning）

        Args:
            graph_emb1: [B, D] 第一组图嵌入，B = batch size
            graph_emb2: [B, D] 第二组图嵌入（增强后的图）
            temperature: 温度参数
        Returns:
            loss: 对比学习损失
        """
        # 1. 归一化
        z1 = F.normalize(graph_emb1, dim=-1)
        z2 = F.normalize(graph_emb2, dim=-1)

        # 2. 相似度矩阵 [B, B]
        sim_matrix = torch.mm(z1, z2.T)

        # 3. 正样本对：对角线是同一图的不同增强
        labels = torch.arange(z1.size(0), device=z1.device)

        # 4. 对比学习 loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = (loss_fn(sim_matrix / temperature, labels) +
                loss_fn(sim_matrix.T / temperature, labels)) / 2

        return loss

    def forward(self, sample,HG_up,HG_pu,uid2col,U_I,uid2idx):
        # Process input sample

        short_term_sequence = sample[-1]
        target = short_term_sequence[0][0, -1]
        user_id = short_term_sequence[0][2, 0]

        # Long-term
        col_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_col) + self.b_gate_col))

        U_gate_pois_embs = torch.multiply(self.UI_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.UI_embedding.weight[:-1],
                                                                       self.w_gate_U) + self.b_gate_U))


        hg_pois_embs = self.mv_hconv_network(col_gate_pois_embs, HG_pu, HG_up)
        hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=-1)

        noise = torch.randn_like(hg_pois_embs) * 0.1  # 高斯噪声
        hg_pois_embs_aug = hg_pois_embs + noise
        level_loss = self.graph_level_contrastive_loss(hg_pois_embs, hg_pois_embs_aug)

        hg_structural_users_embs = torch.sparse.mm(HG_pu, hg_pois_embs)  # [U, d]
        user = uid2col[user_id.item()]
        hg_users_embs = hg_structural_users_embs[user]  # [BS, d]
        user_cl = hg_structural_users_embs[9:]

        user_embs = self.pp_ConvNetwork(U_gate_pois_embs,U_I)
        user_embs = F.normalize(user_embs, p=2, dim=-1)
        noise = torch.randn_like(user_embs) * 0.1  # 高斯噪声
        user_embs_aug = user_embs + noise
        level_loss1 = self.graph_level_contrastive_loss(user_embs, user_embs_aug)

        poi_embs = user_embs[42:]  # 取后 num_pois 个 # 42  764 786
        user_UI = uid2idx[user_id.item()]
        users_emb = user_embs[user_UI]  # [BS, d]
        user_c = user_embs[:42]


        final_p = self.gate1(hg_pois_embs,poi_embs)
        #final_p = poi_embs + hg_pois_embs
        #final_u = self.a * users_emb + (1-self.a) * hg_users_embs
        final_u = users_emb + hg_users_embs
        output= final_p @ final_u.T

        label = torch.unsqueeze(target, 0)
        pred = torch.unsqueeze(output, 0)

        ssl_p = self.graph_contrastive_loss(hg_pois_embs,poi_embs)
        ssl = self.graph_contrastive_loss(user_c, user_cl)

        pred_loss = self.loss_func(pred, label)
        loss = pred_loss  + 1 * (level_loss1 + level_loss) + 1 * (ssl_p + ssl)
        return loss, output

    def predict(self, sample,L,L_PU,uid2col,U_I,uid2idx):
        _, pred_raw = self.forward(sample,L,L_PU,uid2col,U_I,uid2idx)
        ranking = torch.sort(pred_raw, descending=True)[1]
        target = sample[-1][0][0, -1]

        return ranking, target
