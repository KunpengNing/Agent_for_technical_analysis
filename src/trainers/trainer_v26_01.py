import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 将项目根目录加入到系统路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.database.connector import LocalShareClient
from src.models.agent_model_v26_01 import DailyKronosTokenizer, DailyKronos


class DailyStockDataset(Dataset):
    """
    日线级别股票数据集。
    
    该数据集会在初始化时使用 LocalShareClient 拉取数据库中的股票列表，
    并提取出指定长度 (seq_len) 的滑动切片。
    每个切片包含 `open`, `high`, `low`, `close`, `amount` 5个维度。
    时间维度不参与具体的值输入，而是通过 [0, 1, ..., seq_len-1] 的序列传入以供相对位置编码。
    股票代码也不参与训练。
    """
    
    def __init__(self, seq_len=256, max_stocks=None):
        """
        初始化数据集，预加载并切片数据。
        
        Args:
            seq_len (int): 序列切片长度，默认为 256 (240 训练 + 16 预测)。
            max_stocks (int): 最大加载股票数量，用于调试时限制内存。如果为 None，则加载所有可用股票。
        """
        self.seq_len = seq_len
        self.client = LocalShareClient()
        
        print("正在从数据库获取可用股票列表...")
        stock_list = self.client.get_stock_list()
        if max_stocks is not None:
            stock_list = stock_list[:max_stocks]
            
        print(f"计划加载 {len(stock_list)} 只股票的数据...")
        
        self.samples = []
        
        for idx, symbol in enumerate(stock_list):
            if idx % 100 == 0 and idx > 0:
                print(f"已处理 {idx} 只股票...")
                
            df = self.client.get_daily_quote(symbol=symbol, adjust="no_adj")
            
            if df is None or len(df) < self.seq_len:
                continue
            
            # 提取5维特征
            # 'turnover' 暂时不加入，只保留量、价的 5 个维度
            features = df[['open', 'high', 'low', 'close', 'amount']].values.astype(np.float32)
            features = np.nan_to_num(features) # 处理可能存在的 NaN
            
            # 滑动窗口切片
            for i in range(len(features) - self.seq_len + 1):
                window = features[i : i + self.seq_len]
                
                # Z-score 局部标准化：让模型专注于形态特征而非绝对价格
                mean = np.mean(window, axis=0)
                std = np.std(window, axis=0) + 1e-8
                window_norm = (window - mean) / std
                
                self.samples.append(window_norm)
                
        print(f"数据集加载完毕，共生成 {len(self.samples)} 个有效训练切片。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取单个训练样本。
        
        Returns:
            x (torch.Tensor): 形状为 [seq_len, 5] 的标准化特征序列。
            stamp (torch.Tensor): 形状为 [seq_len] 的相对时间序列 [0, 1, ..., seq_len-1]。
        """
        x = torch.tensor(self.samples[index], dtype=torch.float32)
        # 时间序列仅为相对顺序
        stamp = torch.arange(self.seq_len, dtype=torch.long)
        return x, stamp


class DailySequentialTrainer:
    """
    日线级别 Kronos 模型的两阶段顺序训练器。
    
    包含两个阶段：
    1. 训练 DailyKronosTokenizer：使其能够将连续的 5 维特征编码为离散的 token 并解码重建。
    2. 训练 DailyKronos (Base Model)：在 Tokenizer 冻结的情况下，使用自回归方式预测未来的 token。
    
    所有的权重会保存在指定的路径下。
    """
    
    def __init__(self, seq_len=256, batch_size=32, max_stocks=100, device=None):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型架构
        print("初始化 DailyKronosTokenizer (d_in=5)...")
        self.tokenizer = DailyKronosTokenizer(d_in=5).to(self.device)
        
        print("初始化 DailyKronos Base Model...")
        self.model = DailyKronos(
            s1_bits=self.tokenizer.s1_bits, 
            s2_bits=self.tokenizer.s2_bits, 
            n_layers=12, 
            d_model=832, 
            n_heads=16, 
            ff_dim=2048, 
            ffn_dropout_p=0.1, 
            attn_dropout_p=0.0, 
            resid_dropout_p=0.1, 
            token_dropout_p=0.0
        ).to(self.device)
        
        # 初始化数据加载器
        self.dataset = DailyStockDataset(seq_len=seq_len, max_stocks=max_stocks)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        # 设置保存路径
        self.save_dir = os.path.join(project_root, "model_parameter", "parameters_v26.01")
        self.tokenizer_save_path = os.path.join(self.save_dir, "tokenizer.pt")
        self.basemodel_save_path = os.path.join(self.save_dir, "basemodel.pt")
        os.makedirs(self.save_dir, exist_ok=True)

    def train_tokenizer(self, epochs=5, lr=1e-4):
        """
        第一阶段：训练 Tokenizer。
        通过均方误差 (MSE) 和 BSQ 损失 (BSQ Loss) 联合优化，使 Tokenizer 能够重构序列。
        """
        print("\n" + "="*50)
        print("开始阶段 1: 训练 DailyKronosTokenizer")
        print("="*50)
        
        optimizer = torch.optim.AdamW(self.tokenizer.parameters(), lr=lr)
        self.tokenizer.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            start_time = time.time()
            
            for batch_idx, (x, _) in enumerate(self.dataloader):
                x = x.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                (z_pre, z), bsq_loss, quantized, z_indices = self.tokenizer(x)
                
                # 重建损失 (MSE)
                recon_loss = F.mse_loss(z, x)
                
                # 联合损失
                loss = recon_loss + bsq_loss
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.dataloader)} | "
                          f"Loss: {loss.item():.4f} | Recon Loss: {recon_loss.item():.4f}")
            
            avg_loss = total_loss / len(self.dataloader)
            avg_recon = total_recon_loss / len(self.dataloader)
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1} 结束 | 平均 Loss: {avg_loss:.4f} | 平均 Recon Loss: {avg_recon:.4f} | 耗时: {elapsed:.2f}秒")
            
        # 保存 Tokenizer
        torch.save(self.tokenizer.state_dict(), self.tokenizer_save_path)
        print(f"Tokenizer 权重已保存至: {self.tokenizer_save_path}")

    def train_basemodel(self, epochs=10, lr=1e-4):
        """
        第二阶段：训练 Base Model (DailyKronos)。
        在冻结的 Tokenizer 辅助下，将连续特征序列编码为离散 tokens，
        然后训练 Base Model 自回归地预测下一个时间步的 token (s1 和 s2)。
        """
        print("\n" + "="*50)
        print("开始阶段 2: 训练 DailyKronos (Base Model)")
        print("="*50)
        
        # 冻结 Tokenizer
        self.tokenizer.eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            start_time = time.time()
            
            for batch_idx, (x, stamp) in enumerate(self.dataloader):
                x = x.to(self.device)
                stamp = stamp.to(self.device)
                
                optimizer.zero_grad()
                
                # 1. 使用 Tokenizer 将连续序列编码为离散的双层 tokens
                with torch.no_grad():
                    z_indices = self.tokenizer.encode(x, half=True)
                    s1_ids, s2_ids = z_indices[0], z_indices[1]
                
                # 2. 构造自回归的输入和目标 (Teacher Forcing)
                # 输入序列: [0, 1, ..., seq_len-2]
                # 目标序列: [1, 2, ..., seq_len-1]
                s1_input = s1_ids[:, :-1]
                s2_input = s2_ids[:, :-1]
                stamp_input = stamp[:, :-1]
                
                s1_target = s1_ids[:, 1:]
                s2_target = s2_ids[:, 1:]
                
                # 3. 前向传播模型
                s1_logits, s2_logits = self.model(
                    s1_input, s2_input, 
                    stamp=stamp_input, 
                    use_teacher_forcing=True, 
                    s1_targets=s1_target
                )
                
                # 4. 计算交叉熵损失
                loss, ce_s1, ce_s2 = self.model.head.compute_loss(s1_logits, s2_logits, s1_target, s2_target)
                
                # 5. 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.dataloader)} | "
                          f"Loss: {loss.item():.4f} (s1: {ce_s1.item():.4f}, s2: {ce_s2.item():.4f})")
            
            avg_loss = total_loss / len(self.dataloader)
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1} 结束 | 平均 Loss: {avg_loss:.4f} | 耗时: {elapsed:.2f}秒")
            
        # 保存 Base Model
        torch.save(self.model.state_dict(), self.basemodel_save_path)
        print(f"Base Model 权重已保存至: {self.basemodel_save_path}")

    def run_training(self):
        """执行完整顺序训练"""
        self.train_tokenizer()
        self.train_basemodel()
        print("\n所有训练任务已圆满完成！")


if __name__ == "__main__":
    # 执行测试或正式训练
    # 这里我们加载少量的股票来验证 pipeline 能够正常跑通
    trainer = DailySequentialTrainer(seq_len=256, batch_size=16, max_stocks=10)
    trainer.run_training()
