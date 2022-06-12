import pandas as pd
from src.models.lstm_model import LSTM
from src.data.dataset import create_loaders
import matplotlib.pyplot as plt
import torch


def init_objects(job_config):
    dataframe = pd.read_csv(job_config["dataset"]["path"])
    train_dl, val_dl = create_loaders(dataframe=dataframe,
                                      split_ratio=job_config["dataset"]["split_ratio"],
                                      batch_size=job_config["dataset"]["batch_size"],
                                      seq_length=job_config["dataset"]["seq_length"])
    print("Dataloaders Generated...")
    model = LSTM(num_classes=job_config["model"]["num_classes"],
                 input_size=job_config["model"]["input_size"],
                 hidden_size=job_config["model"]["hidden_size"],
                 num_layers=job_config["model"]["num_layers"])
    print("Model Generated...")
    print(model)
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=job_config["optimizer"]["lr"],
                             betas=tuple(job_config["optimizer"]["betas"]))
    print("Optimizer Generated...")
    loss_fn = torch.nn.MSELoss()

    return (train_dl, val_dl), model, optim, loss_fn


def save_best_model(curr_loss, best_loss, model, optim):
    if curr_loss <= best_loss:
        weights = {
            "model": model.state_dict(),
            "optim": optim.state_dict()
        }
        torch.save(weights, "best_model.pt")
        print("Model Updated...")
    else:
        print("Model didn't Updated...")
    best_loss = min(curr_loss, best_loss)
    print(f"Current best loss : {best_loss}")
    return best_loss


def save_training_curve(train_array, val_array):
    plt.figure(figsize=(20, 6))
    plt.plot(train_array, label="training_loss")
    plt.plot(val_array, label="validation_loss")
    plt.legend()
    plt.savefig("training_metrics.png")
    plt.close()

