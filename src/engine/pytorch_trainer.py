import yaml
import torch
import numpy as np
from src.utils.training import (init_objects,
                                save_best_model_on_loss,
                                save_training_curve,
                                update_metric_dict)
from src.utils.mlflow_logger import log_pt_models
from src.utils.load_checkpoint import load_best_model
from src.utils.metrics import metrics_


def trainer(model, train_dl, val_dl, loss_fn, optim, job_config):
    training_hp = job_config["training_hp"]
    epochs = training_hp["num_epochs"]
    log_index = training_hp["log_index"]
    best_train_loss = torch.inf
    best_val_loss = torch.inf
    train_loss_array, val_loss_array = [], []
    train_metric_results = {k: [] for k, _ in metrics_.items()}
    val_metric_results = {k: [] for k, _ in metrics_.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device found : {device}")
    model.to(device)
    print('Model loaded on device...')
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} :")
        train_epoch_metric_results = {k: [] for k, _ in metrics_.items()}
        val_epoch_metric_results = {k: [] for k, _ in metrics_.items()}
        train_loss = val_loss = 0.0
        model.train()
        for index, (x, y) in enumerate(train_dl):
            if device != "cpu":
                x = x.cuda()
                y = y.cuda()
            op = model(x, device)
            train_epoch_metric_results = update_metric_dict(y, op, metrics_, train_epoch_metric_results)
            curr_loss = loss_fn(op, y)
            if index % log_index == 0:
                print(f"Step {index} Loss : {'%.6f' % curr_loss.item()}")
            train_loss += curr_loss.item()
            curr_loss.backward()
            optim.step()
        model.eval()
        for x, y in val_dl:
            if device != "cpu":
                x = x.cuda()
                y = y.cuda()
            op = model(x, device)
            val_epoch_metric_results = update_metric_dict(y, op, metrics_, val_epoch_metric_results)
            curr_loss = loss_fn(op, y)
            val_loss += curr_loss.item()
        print(f"Train Loss : {'%.6f' % train_loss} || Validation Loss : {'%.6f' % val_loss}")
        train_res = {k: np.mean(v) for k, v in train_epoch_metric_results.items()}
        val_res = {k: sum(v) for k, v in val_epoch_metric_results.items()}
        print(f"Train Metric Results : {train_res}")
        print(f"Validation Metric Results : {val_res}")
        for tot, epo in zip([train_metric_results, val_metric_results],
                            [train_res, val_res]):
            for name, val in epo.items():
                tot[name].append(val)
        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)
        best_train_loss, best_val_loss = save_best_model_on_loss(curr_losses=(train_loss, val_loss),
                                                                 best_losses=(best_train_loss, best_val_loss),
                                                                 model=model,
                                                                 optim=optim)
        print("\n")
    results = {
        "training_mse": best_train_loss,
        "validation_mse": best_val_loss
    }
    best_model = load_best_model(job_config["model"])
    log_pt_models(model=best_model,
                  hparams=job_config,
                  results=results)
    train_metric_results["mse"] = train_loss_array
    val_metric_results["mse"] = val_loss_array
    save_training_curve(train_metric_results, val_metric_results)
    print("Training Completed...")


def main():
    job_config_path = "config/pt_training.yaml"
    with open(job_config_path, "r") as f:
        job_config = yaml.safe_load(f)
        f.close()
    print("Configuration Loaded...")
    print(job_config)
    (train_dl, val_dl), lstm_model, optim, loss_fn = init_objects(job_config)
    trainer(model=lstm_model,
            train_dl=train_dl,
            val_dl=val_dl,
            loss_fn=loss_fn,
            optim=optim,
            job_config=job_config)


if __name__ == "__main__":
    main()
