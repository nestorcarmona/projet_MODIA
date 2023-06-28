# deep learning
import pandas as pd
import torch

# data
import numpy as np

# random
import random

# os
import argparse
from pathlib import Path

# NCF
from NCF import NCF

def seed_everything(seed_value: int):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # Numpy
    torch.manual_seed(seed_value) # PyTorch
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Make a argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/weight.pth")
    parser.add_argument("--test_script_path", type=str, default="./data/test_script.csv")

    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    test_script_path = Path(args.test_script_path)

    # Check the model path
    if not model_path.exists():
        raise ValueError("model path is not exists")
    # Check the test script path
    if not test_script_path.exists():
        raise ValueError("test script path is not exists")

    # Seed everything
    seed_everything(42)

    # Load the NCF model
    model = NCF(
        n_users=25077,
        n_items=178264,
        n_factors=16,
    )
    # Load the weight
    model.load_state_dict(torch.load(model_path))
    # Set the model to eval mode
    model.eval()

    # Load the test script
    test_script = pd.read_csv(test_script_path)
    data = test_script[['u', 'i', 'rating']].to_numpy(dtype=np.int64)
    users_test = data[:, 0]
    items_test = data[:, 1]
    ratings_test = data[:, 2]

    # Make the test script to torch tensor
    users_test = torch.from_numpy(users_test)
    items_test = torch.from_numpy(items_test)
    ratings_test = torch.from_numpy(ratings_test)

    # Predict the test script
    with torch.no_grad():
        ratings_predict = model.forward(user=users_test, item=items_test)*5

    # print the result
    for i in range(len(ratings_test)):
        print(f"ground truth: {ratings_test[i]}, predicted: {ratings_predict[i].item():.3f}")

    

