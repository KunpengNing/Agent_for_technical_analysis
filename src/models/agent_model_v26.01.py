import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
import numpy as np

# 将项目根目录加入到系统路径中，以便能够引用 _Kronos 文件夹下的模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from _Kronos.models.module import (
    TransformerBlock, 
    BSQuantizer, 
    HierarchicalEmbedding, 
    RMSNorm, 
    DependencyAwareLayer, 
    DualHead,
    FixedEmbedding
)


class RelativeTemporalEmbedding(nn.Module):
    """
    相对位置时间编码层。
    与原版的高频时间特征 (minute, hour, weekday等) 不同，
    为了防止模型学到特定的牛市、熊市区间，我们将输入的时间戳简化为纯粹的相对序列索引 [0, 1, 2, ...]。
    
    Args:
        d_model (int): 模型的隐藏层维度。
        max_len (int): 序列的最大长度，默认为 5000。
    """
    def __init__(self, d_model, max_len=5000):
        super(RelativeTemporalEmbedding, self).__init__()
        self.pos_embed = FixedEmbedding(max_len, d_model)

    def forward(self, stamp):
        """
        前向传播函数。
        
        Args:
            stamp (torch.Tensor): 形状为 [batch_size, seq_len] 的时间戳序列。
                                  内容应为 [0, 1, 2, ...] 的相对位置索引。
                                  
        Returns:
            torch.Tensor: 形状为 [batch_size, seq_len, d_model] 的时间编码。
        """
        # stamp 应该是 [0, 1, 2, ..., seq_len-1]
        # 使用 FixedEmbedding 提取相对位置的表征
        return self.pos_embed(stamp.long())


class DailyKronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    日线级别特征的分词器 (Tokenizer)。
    继承自原版的 KronosTokenizer 思路，但将默认输入维度改为 5 (开、高、低、收、成交额)。
    它包含 Encoder 和 Decoder 两个部分，中间通过 Binary Spherical Quantization (BSQ) 将连续特征离散化。
    """

    def __init__(self, d_in=5, d_model=256, n_heads=4, ff_dim=512, n_enc_layers=4, n_dec_layers=4, 
                 ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, 
                 s1_bits=10, s2_bits=10, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits
        
        # 输入和输出的投影层
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # 编码器 (Encoder) Transformer Block 序列
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])
        
        # 解码器 (Decoder) Transformer Block 序列
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])
        
        # 量化前后的映射层
        self.quant_embed = nn.Linear(in_features=self.d_model, out_features=self.codebook_dim) 
        self.post_quant_embed_pre = nn.Linear(in_features=self.s1_bits, out_features=self.d_model) 
        self.post_quant_embed = nn.Linear(in_features=self.codebook_dim, out_features=self.d_model) 
        
        # 核心的 BSQuantizer
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size)

    def forward(self, x):
        """
        前向传播计算。
        
        Args:
            x (torch.Tensor): 形状为 [batch_size, seq_len, d_in] 的输入连续特征。
            
        Returns:
            tuple: 包含重构输出 (z_pre, z)，BSQ的loss，量化向量以及离散索引的元组。
        """
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z) # 投影到 codebook 维度

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        # 提取高位特征 (pre part - s1 bits)
        quantized_pre = quantized[:, :, :self.s1_bits] 
        z_pre = self.post_quant_embed_pre(quantized_pre)

        # 完整特征
        z = self.post_quant_embed(quantized)

        # 高位特征的解码
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # 完整特征的解码
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        """将索引还原回二进制比特向量"""
        if half:
            x1, x2 = x[0], x[1]
            mask = 2 ** torch.arange(self.codebook_dim//2, device=x1.device, dtype=torch.long)
            x1 = (x1.unsqueeze(-1) & mask) != 0 
            x2 = (x2.unsqueeze(-1) & mask) != 0 
            x = torch.cat([x1, x2], dim=-1)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long) 
            x = (x.unsqueeze(-1) & mask) != 0 

        x = x.float() * 2 - 1 
        q_scale = 1. / (self.codebook_dim ** 0.5)
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        """将连续输入序列编码为离散的 token 索引"""
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x, half=False):
        """将离散的 token 索引解码回连续序列特征"""
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class DailyKronos(nn.Module, PyTorchModelHubMixin):
    """
    日线级别自回归预测模型的主体架构。
    接收由 Tokenizer 提取的离散 Tokens (s1_ids, s2_ids)，
    并结合相对时间编码 (RelativeTemporalEmbedding)，预测未来的 Tokens。
    """

    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim, 
                 ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te=False):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(token_dropout_p)
        
        # 分层 Embedding 结构处理粗粒度和细粒度的 token
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        
        # 使用仅基于相对位置序列的 Daily 时间编码层
        self.time_emb = RelativeTemporalEmbedding(self.d_model)
        
        # 自回归 Transformer Blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(self.n_layers)
        ])
        
        self.norm = RMSNorm(self.d_model)
        # 依赖层：在预测细粒度 token 时条件化于对应的粗粒度 token
        self.dep_layer = DependencyAwareLayer(self.d_model)
        # 双重预测头，分别预测 s1 (粗粒度) 和 s2 (细粒度) token
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化函数"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        """
        前向传播函数。
        
        Args:
            s1_ids (torch.Tensor): 粗粒度 token ID [batch_size, seq_len]
            s2_ids (torch.Tensor): 细粒度 token ID [batch_size, seq_len]
            stamp (torch.Tensor): 相对时间戳 [batch_size, seq_len]，形如 [0, 1, 2, ...]
            padding_mask (torch.Tensor): Padding 掩码
            use_teacher_forcing (bool): 是否使用 teacher forcing 来训练 s2
            s1_targets (torch.Tensor): s1 的目标 ID
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: s1_logits, s2_logits
        """
        x = self.embedding([s1_ids, s2_ids])
        
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
            
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        # 预测 s1 (粗粒度) token
        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        # 基于 s1 的预测结果进行条件化，进而预测 s2 (细粒度) token
        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        
        return s1_logits, s2_logits
