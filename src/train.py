import argparse
from pathlib import Path
import yaml
# import mlflow
import pytorch_lightning as pl
# from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data.dataloader import get_dataloaders
from lightning_modules.base import BaseModule
# from utils.mlflow_utils import init_mlflow_experiment


def main(args):
    # Load configurations file
    conf_file = Path("./config") / args.config
    with open(conf_file) as f:
        config = yaml.safe_load(f)

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config)
    module = BaseModule(config)

    # # Initialize MLflow experiment
    # init_mlflow_experiment(config)

    # # Set up MLFlow Logger
    # mlflow_logger = MLFlowLogger(
    #     experiment_name=config['mlflow_experiment_name'],
    #     tracking_uri='file:./mlruns'
    # )

    # Set up Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=config['model_name'],
        save_top_k=1,
        mode='min'
    )

    # Set up the Trainer
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        # logger=mlflow_logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(module, train_dataloader, val_dataloader)

    # Evaluate on test dataset
    test_result = trainer.test(module, test_dataloader)
    print(test_result[0]['test_acc'])

    # # Log test results to MLflow
    # mlflow.log_metrics({'test_loss': test_result[0]['test_loss'],
    #                     'test_acc': test_result[0]['test_acc']})

    # # End the MLflow experiment
    # mlflow.end_run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
