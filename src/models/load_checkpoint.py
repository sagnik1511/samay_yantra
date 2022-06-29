import torch
from src.models.lstm_model import LSTM


def load_model(path, model_param, device="cpu"):
    chkp = torch.load(path, map_location=device)
    del model_param["checkpoint_path"]
    model = LSTM(**model_param)
    model.load_state_dict(chkp["model"])

    return model
