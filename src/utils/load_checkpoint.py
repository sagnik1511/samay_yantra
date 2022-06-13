import torch
from src.models.lstm_model import LSTM


def load_best_model(model_config):
    path = "best_model.pt"
    chkp = torch.load(path)
    model = LSTM(**model_config)
    model.load_state_dict(chkp["model"])

    return model
