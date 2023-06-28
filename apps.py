import pandas as pd
import torch
from NCF import NCF

if __name__ == "__main__":
    # Define the device with mps if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # Load the test data (example)
    test_script = pd.read_csv('../data/test_script.csv')
    
    # Load the model
    model = NCF(n_users=25077, n_items=178264, n_factors=16).to(device)
    model.load_state_dict(torch.load('../model/weights.pth'))
    model.eval()

    # Define the data
    X_u = torch.from_numpy(test_script['u'].to_numpy()).to(device)
    X_i = torch.from_numpy(test_script['i'].to_numpy()).to(device)
    y = test_script['rating'].to_numpy()

    # Predict
    with torch.no_grad():
        y_pred = model(X_u, X_i)
    y_pred = y_pred.cpu().detach().numpy()

    # Save the prediction
    test_script['rating'] = y_pred
    


