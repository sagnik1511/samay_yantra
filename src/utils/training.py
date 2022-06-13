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
                             lr=job_config["optimizer"]["lr"])
    print("Optimizer Generated...")
    loss_fn = torch.nn.MSELoss()

    return (train_dl, val_dl), model, optim, loss_fn


def save_best_model_on_loss(curr_losses, best_losses, model, optim, track_on="validation"):
    assert track_on in ["training", "validation"]
    curr_train_loss, curr_val_loss = curr_losses
    best_train_loss, best_val_loss = best_losses
    if track_on == "training":
        flag = save_best_model(curr_loss=curr_train_loss,
                               best_loss=best_train_loss,
                               model=model,
                               optim=optim)
    else:
        flag = save_best_model(curr_loss=curr_val_loss,
                               best_loss=best_val_loss,
                               model=model,
                               optim=optim)
    if flag:
        return curr_losses
    else:
        return best_losses


def save_best_model(curr_loss, best_loss, model, optim):
    if curr_loss <= best_loss:
        weights = {
            "model": model.state_dict(),
            "optim": optim.state_dict()
        }
        torch.save(weights, "best_model.pt")
        print("Model Updated...")
        print(f"Current best loss : {'%.6f'%curr_loss}")
        return True
    else:
        print("Model didn't Updated...")
        print(f"Current best loss : {'%.6f' % best_loss}")
        return False


def save_training_curve(train_array, val_array):
    plt.figure(figsize=(20, 6))
    plt.plot(train_array, label="training_loss")
    plt.plot(val_array, label="validation_loss")
    plt.legend()
    plt.savefig("results/training_metrics.png")
    plt.close()

