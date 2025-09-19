import mlflow
import mlflow.pytorch

def train_and_log(config):
    mlflow.set_tracking_uri(config.logging.mlflow_tracking_uri)
    mlflow.set_experiment(config.logging.experiment_name)

    with mlflow.start_run():
        # log hyperparams
        mlflow.log_params({
            "cnn": config.model.cnn,
            "transformer": config.model.transformer,
            "fusion": config.model.fusion,
            "lr": config.training.lr,
            "batch_size": config.training.batch_size
        })

        model = HybridModel(config.model)  # from models/hybrid_model.py
        # train ...
        val_acc = 0.85  # placeholder
        
        mlflow.log_metric("val_acc", val_acc)
        mlflow.pytorch.log_model(model, "model")
