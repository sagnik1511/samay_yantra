from src.engine.predictor import fetch_index, predict_results
from src.models.load_checkpoint import load_model
import yaml


def predict(user_input, config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        f.close()
    print(config)
    model = load_model(config["model"]["checkpoint_path"], config["model"])
    index = fetch_index(user_input, config["dataset"]["seq_length"], config["dataset"]["path"])
    print(f"index : {index}")
    ans = predict_results(config["dataset"]["path"], index, model, config["dataset"]["seq_length"])

    return ans


if __name__ == "__main__":
    user_inp = "2021-02-01 08:50:00"
    ans = predict(user_inp, "config/predict.yaml")
    print(ans)


