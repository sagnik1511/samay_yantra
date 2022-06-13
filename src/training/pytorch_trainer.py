import yaml
import torch
from src.utils.training import init_objects, save_best_model_on_loss, save_training_curve
from src.utils.mlflow_logger import log_pt_models
from src.utils.load_checkpoint import load_best_model


def trainer(model, train_dl, val_dl, loss_fn, optim, job_config):
    training_hp = job_config["training_hp"]
    epochs = training_hp["num_epochs"]
    log_index = training_hp["log_index"]
    best_train_loss = torch.inf
    best_val_loss = torch.inf
    train_loss_array, val_loss_array = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device found : {device}")
    model.to(device)
    print('Model loaded on device...')
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} :")
        train_loss = val_loss = 0.0
        model.train()
        for index, (x, y) in enumerate(train_dl):
            if device != "cpu":
                x = x.cuda()
                y = y.cuda()
            op = model(x, device)
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
            curr_loss = loss_fn(op, y)
            val_loss += curr_loss.item()
        print(f"Train Loss : {'%.6f' % train_loss} || Validation Loss : {'%.6f' % val_loss}")
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
    save_training_curve(train_loss_array, val_loss_array)
    print("Training Completed...")


def main():
    job_config_path = "config/pt_cpu_training.yaml"
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
