import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import torch


if __name__ == "__main__":
    print("Hello World")
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(x.cuda())
        
        
    