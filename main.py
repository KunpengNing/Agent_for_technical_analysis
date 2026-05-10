from src.trainers.trainer_v26_01 import TokenizerTrainer, PredictorTrainer

if __name__ == "__main__":
    # 示例用法
    
    # 1. 训练 Tokenizer
    t_trainer = TokenizerTrainer(max_stocks=10)
    t_trainer.train(epochs=2)
    
    # # 2. 训练 Predictor
    # p_trainer = PredictorTrainer(max_stocks=10)
    # p_trainer.train(epochs=2)