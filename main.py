from src.trainers.trainer_v26_01 import TokenizerTrainer, PredictorTrainer

if __name__ == "__main__":
    # 示例用法
    
    # 1. 训练 Tokenizer
    t_trainer = TokenizerTrainer(seq_len=256, batch_size=32, num_samples=1024)
    t_trainer.train(epochs=30)