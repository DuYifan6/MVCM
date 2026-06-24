#model.py
import torch
import torch.nn as nn
from transformers import BertModel


class ConsistencyAttention(nn.Module):
    def __init__(self, dim: int, w_scalar_init: float = 3.0):
        super().__init__()
        self.scale = dim ** -0.5
        self.w_scalar = nn.Parameter(torch.tensor(w_scalar_init, dtype=torch.float))

    def forward(self, Q, K, V, mvc_matrix):
        """
        Q,K,V: (B, L, dim), L=3
        mvc_matrix: (B, L, L)
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,3,3)
        mvc = mvc_matrix.to(attn_scores.device).type(attn_scores.dtype)
        attn_scores = attn_scores + self.w_scalar * mvc
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, V)
        return out, attn


class MultiViewPooling(nn.Module):
    """
    自适应多视图池化：对每个视图的输出打一个标量分数，再 softmax 做加权求和。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, views):
        """
        views: (B, L, H)
        return:
            pooled: (B, H)
            weights: (B, L)
        """
        logits = self.score(views)              # (B,L,1)
        weights = torch.softmax(logits, dim=1)  # (B,L,1)
        pooled = (weights * views).sum(dim=1)   # (B,H)
        return pooled, weights.squeeze(-1)      # (B,H), (B,L)


class MVCTModel(nn.Module):
    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        w_scalar_init=3.0,
        dropout_p=0.1,
        mvc_weights=(0.5, 0.2, 0.15, 0.15),  # S, F, L, Local
    ):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_model_name)
        hidden = self.encoder.config.hidden_size

        self.consist_attn = ConsistencyAttention(hidden, w_scalar_init)
        self.pool = MultiViewPooling(hidden)

        self.fc = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout_p)

        self.head_fake = nn.Linear(hidden, 2)
        self.head_ai   = nn.Linear(hidden, 2)

        self.last_attn = None
        self.last_view_pool = None

        # ⭐ MVCM 通道权重
        self.register_buffer(
            "mvc_weights",
            torch.tensor(mvc_weights, dtype=torch.float)
        )

    def encode_cls(self, field):
        out = self.encoder(
            input_ids=field["input_ids"],
            attention_mask=field["attention_mask"],
        )
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

    def _prepare_mvc(self, mvc_tensor, batch_size: int, device):
        """
        将 MVCM (可能是 3 通道或 4 通道) 按权重融合为 (B,3,3) 的单通道矩阵，
        用于 ConsistencyAttention 里和 QK 结果相加。

        支持的输入形状：
          - (B, 3, 3, C)
          - (3, 3, C)
          - (3, 3)
        其中 C <= len(self.mvc_weights) (一般是 3 或 4)
        """
        mvc = mvc_tensor.to(device)

        # ---------- case 1: batched (B,3,3,C) ----------
        if mvc.dim() == 4:
            B, h1, h2, C = mvc.shape
            assert h1 == 3 and h2 == 3, f"Expect (B,3,3,C), got {mvc.shape}"

            w = self.mvc_weights[:C].to(device).view(1, 1, 1, C)  # (1,1,1,C)
            mvc = (mvc * w).sum(dim=-1)  # (B,3,3)

        # ---------- case 2: 单样本带通道 (3,3,C) ----------
        elif mvc.dim() == 3:
            if mvc.size(0) == 3 and mvc.size(1) == 3:
                C = mvc.size(2)
                if C > 1:
                    mvc = mvc.unsqueeze(0)  # (1,3,3,C)
                    w = self.mvc_weights[:C].to(device).view(1, 1, 1, C)
                    mvc = (mvc * w).sum(dim=-1)  # (1,3,3)
                else:
                    mvc = mvc.squeeze(-1).unsqueeze(0)  # (1,3,3)
            else:
                raise ValueError(f"Unexpected 3D MVC shape: {mvc.shape}")

        # ---------- case 3: 单样本已是 (3,3) ----------
        elif mvc.dim() == 2 and mvc.size(0) == 3 and mvc.size(1) == 3:
            mvc = mvc.unsqueeze(0)  # (1,3,3)

        else:
            raise ValueError(f"Unexpected MVC tensor shape: {mvc.shape}")

        # 如果只有 1 个 MVC，但 batch_size > 1，就广播
        if mvc.size(0) == 1 and batch_size > 1:
            mvc = mvc.expand(batch_size, -1, -1)  # (B,3,3)

        return mvc  # (B,3,3)

    def forward(self, title_field, text_field, desc_field, mvc_combined):
        h_t = self.encode_cls(title_field)  # (B,H)
        h_x = self.encode_cls(text_field)
        h_d = self.encode_cls(desc_field)

        B = h_t.size(0)
        device = h_t.device

        H = torch.stack([h_t, h_x, h_d], dim=1)  # (B,3,H)

        mvc_mat = self._prepare_mvc(mvc_combined, batch_size=B, device=device)

        out, attn = self.consist_attn(H, H, H, mvc_mat)  # (B,3,H), (B,3,3)

        # 残差
        out = out + H

        # 训练时对 view-attn 做一些 dropout，防止视图塌缩
        if self.training:
            keep_prob = 0.8
            mask = (torch.rand(attn.shape, device=attn.device) < keep_prob).float()
            attn = attn * mask
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-12)
            out = torch.matmul(attn, H) + H

        self.last_attn = attn.detach() if not self.training else attn

        pooled, view_weights = self.pool(out)  # (B,H), (B,3)
        self.last_view_pool = view_weights.detach() if not self.training else view_weights

        feat = torch.relu(self.fc(pooled))
        feat = self.dropout(feat)

        y_fake = self.head_fake(feat)
        y_ai   = self.head_ai(feat)

        return y_fake, y_ai, attn
