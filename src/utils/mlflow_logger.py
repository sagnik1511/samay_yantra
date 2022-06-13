import mlflow
from urllib.parse import urlparse


def log_pt_models(model, hparams, results):
    with mlflow.start_run():
        for key in ["batch_size", "seq_length", "split_ratio"]:
            mlflow.log_param(key, hparams["dataset"][key])
        for key in ["model", "optimizer", "training_hp"]:
            mlflow.log_params(hparams[key])
        mlflow.log_metrics(results)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.pytorch.log_model(model, "model", registered_model_name="LSTM_Model")
        else:
            mlflow.pytorch.log_model(model, "model")
