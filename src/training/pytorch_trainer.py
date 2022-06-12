import yaml
import torch
from src.utils.training import init_objects, save_best_model, save_training_curve


def trainer(model, train_dl, val_dl, loss_fn, optim, training_hp):
    epochs = training_hp["num_epochs"]
    log_index = training_hp["log_index"]
    best_loss = torch.inf
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
        val_loss_array += [val_loss for _ in range(4)]
        best_loss = save_best_model(curr_loss=val_loss,
                                    best_loss=best_loss,
                                    model=model,
                                    optim=optim)
        print("\n")
    save_training_curve(train_loss_array, val_loss_array)
    print("Training Completed...")


def main():
    job_config_path = "config/pt_cpu_training.yaml"
    with open(job_config_path, "r") as f:
        job_config = yaml.safe_load(f)
        f.close()
    print("Configuration Loaded...")
    (train_dl, val_dl), lstm_model, optim, loss_fn = init_objects(job_config)
    trainer(model=lstm_model,
            train_dl=train_dl,
            val_dl=val_dl,
            loss_fn=loss_fn,
            optim=optim,
            training_hp=job_config["training_hp"])


if __name__ == "__main__":
    main()
