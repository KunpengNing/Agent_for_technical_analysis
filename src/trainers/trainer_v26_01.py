import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime, timedelta

# 将项目根目录加入到系统路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.database.connector import LocalShareClient
from src.models.agent_model_v26_01 import Tokenizer, Predictor


class DailyStockDataset(Dataset):
    """
    日线级别股票数据集。
    """
    def __init__(self,
        seq_len: int = 256,
        num_samples: int = 1024,
        start_date: str = "20150101",
        end_date: str = "20250101",
    ):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.start_date = start_date
        self.end_date = end_date
        client = LocalShareClient()

        print(f"[{self.__class__.__name__}] 正在从数据库获取可用股票列表")
        raw_stock_list = client.get_stock_list()

        # 1. 过滤股票 (Mod 7 排除测试集)
        stocks_for_training = []
        stocks_for_test = []
        for s in raw_stock_list:
            # 提取 symbol 中的数字部分 (如 'sh600001' -> 600001)
            num_part = "".join(filter(str.isdigit, s))
            if num_part and int(num_part) % 7 == 0:
                stocks_for_test.append(s)
            else:
                stocks_for_training.append(s)

        print(f"[{self.__class__.__name__}] 获取完成，用于训练的股票数量: {len(stocks_for_training)}, 用于测试的股票数量: {len(stocks_for_test)}")


        # 2. 构建切片元数据池
        self.samples = []
        print(f"[{self.__class__.__name__}] 开始构建切片池，目标样本数量: {self.num_samples}")

        date_format = "%Y%m%d"
        start_datetime = datetime.strptime(self.start_date, date_format)
        end_datetime = datetime.strptime(self.end_date, date_format)
        delta_days = (end_datetime - start_datetime).days
        
        with tqdm(total=self.num_samples, desc="构建数据切片") as pbar:
            while len(self.samples) < self.num_samples:
                stock_symbol = random.choice(stocks_for_training)
                random_days = random.randint(0, delta_days)
                stock_start = start_datetime + timedelta(days=random_days)
                # 预留1.8倍的日历天数，以确保有足够的交易日
                stock_end = stock_start + timedelta(days=int(self.seq_len * 1.8))
                stock_start_str = stock_start.strftime("%Y%m%d")
                stock_end_str = stock_end.strftime("%Y%m%d")
    
                try:
                    df = client.get_daily_quote(stock_symbol, start_date=stock_start_str, end_date=stock_end_str, adjust="back_adj")
                    if df is None or df.empty or df.shape[0] < self.seq_len:
                        continue
                    
                    df = df.iloc[:self.seq_len, :]
                    features = df[['open', 'high', 'low', 'close', 'amount']].values.astype(np.float32)
                    
                    # 过滤包含无效值（如停牌导致的 0 价格/成交量，或者 NaN）的切片
                    if np.isnan(features).any() or (features <= 0).any():
                        continue
                        
                    # --- 数据归一化 (Instance Normalization) ---
                    # 1. 价格通道归一化 (Open, High, Low, Close) - 共享均值和标准差以保持K线形态
                    prices = features[:, :4]
                    p_mean = prices.mean()
                    p_std = prices.std() + 1e-8
                    features[:, :4] = (prices - p_mean) / p_std
                    
                    # 2. 成交量归一化 (Amount) - 差异巨大，先取对数压缩极值，再做标准化
                    amounts = np.log(features[:, 4])
                    a_mean = amounts.mean()
                    a_std = amounts.std() + 1e-8
                    features[:, 4] = (amounts - a_mean) / a_std
                    
                    self.samples.append(features)
                    pbar.update(1)
    
                except Exception as e:
                    # 获取出错直接跳过
                    continue


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = torch.tensor(self.samples[index], dtype=torch.float32)
        stamp = torch.arange(self.seq_len, dtype=torch.long)
        return x, stamp


class BaseTrainer:
    """训练器基类，处理设备、数据加载和保存路径。"""
    def __init__(self, seq_len=256, batch_size=32, num_samples=4096, device=None):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据加载
        self.dataset = DailyStockDataset(seq_len=seq_len, num_samples=num_samples)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        # 路径设置
        self.save_dir = os.path.join(project_root, "model_parameter", "parameters_v26.01")
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)


class TokenizerTrainer(BaseTrainer):
    """
    专门负责训练 Tokenizer 的训练器。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("初始化 Tokenizer (d_in=5)...")
        self.tokenizer = Tokenizer(d_in=5).to(self.device)
        self.save_path = os.path.join(self.save_dir, "tokenizer.pt")

    def save_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.checkpoint_dir, f"tokenizer_epoch_{epoch}.pt")
        torch.save(self.tokenizer.state_dict(), ckpt_path)
        print(f"已保存 Tokenizer Checkpoint: {ckpt_path}")

    def load(self, path=None):
        path = path or self.save_path
        if os.path.exists(path):
            self.tokenizer.load_state_dict(torch.load(path, map_location=self.device))
            print(f"成功加载 Tokenizer 权重: {path}")
            return True
        return False

    def train(self, epochs=5, lr=1e-4, save_freq=1, resume_from=None):
        if resume_from:
            self.load(resume_from)
            
        print("\n" + "="*50)
        print("开始训练 DailyKronosTokenizer")
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
                (z_pre, z), bsq_loss, quantized, z_indices = self.tokenizer(x)
                recon_loss = F.mse_loss(z, x)
                loss = recon_loss + bsq_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.dataloader)} | "
                          f"Loss: {loss.item():.4f} | Recon Loss: {recon_loss.item():.4f}")
            
            avg_loss = total_loss / len(self.dataloader)
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1} 结束 | 平均 Loss: {avg_loss:.4f} | 耗时: {elapsed:.2f}秒")
            
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch + 1)
            
        torch.save(self.tokenizer.state_dict(), self.save_path)
        print(f"Tokenizer 最终权重已保存至: {self.save_path}")


class PredictorTrainer(BaseTrainer):
    """
    专门负责训练 Predictor (Base Model) 的训练器。
    需要一个预训练好的 Tokenizer 来进行特征离散化。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        print("初始化 Tokenizer (仅用于编码)...")
        self.tokenizer = Tokenizer(d_in=5).to(self.device)
        self.tokenizer.eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False
            
        print("初始化 Predictor Model...")
        self.model = Predictor(
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
        
        self.tokenizer_path = os.path.join(self.save_dir, "tokenizer.pt")
        self.save_path = os.path.join(self.save_dir, "basemodel.pt")

    def load_tokenizer(self, path=None):
        path = path or self.tokenizer_path
        if os.path.exists(path):
            self.tokenizer.load_state_dict(torch.load(path, map_location=self.device))
            print(f"成功加载 Tokenizer 权重: {path}")
            return True
        print(f"错误: 未找到 Tokenizer 权重: {path}")
        return False

    def save_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.checkpoint_dir, f"basemodel_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"已保存 Predictor Checkpoint: {ckpt_path}")

    def load(self, path=None):
        path = path or self.save_path
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"成功加载 Predictor 权重: {path}")
            return True
        return False

    def train(self, epochs=10, lr=1e-4, save_freq=1, resume_from=None):
        # 确保 Tokenizer 已就绪
        if not self.load_tokenizer():
            raise RuntimeError("必须先有训练好的 Tokenizer 权重才能训练 Predictor。")

        if resume_from:
            self.load(resume_from)
            
        print("\n" + "="*50)
        print("开始训练 DailyKronos (Predictor)")
        print("="*50)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            start_time = time.time()
            
            for batch_idx, (x, stamp) in enumerate(self.dataloader):
                x = x.to(self.device)
                stamp = stamp.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.no_grad():
                    z_indices = self.tokenizer.encode(x, half=True)
                    s1_ids, s2_ids = z_indices[0], z_indices[1]
                
                s1_input, s2_input, stamp_input = s1_ids[:, :-1], s2_ids[:, :-1], stamp[:, :-1]
                s1_target, s2_target = s1_ids[:, 1:], s2_ids[:, 1:]
                
                s1_logits, s2_logits = self.model(
                    s1_input, s2_input, 
                    stamp=stamp_input, 
                    use_teacher_forcing=True, 
                    s1_targets=s1_target
                )
                
                loss, ce_s1, ce_s2 = self.model.head.compute_loss(s1_logits, s2_logits, s1_target, s2_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.dataloader)} | Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1} 结束 | 平均 Loss: {avg_loss:.4f} | 耗时: {time.time() - start_time:.2f}秒")
            
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch + 1)
            
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Predictor 最终权重已保存至: {self.save_path}")



