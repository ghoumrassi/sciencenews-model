import datetime as dt
import argparse
from pathlib import Path
import yaml
import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler
from torch import set_float32_matmul_precision

from data.dataloader import get_dataloaders
from lightning_modules.base import BaseModule


def main(args):
    """
    Main routine for the model training pipeline.

    Args:
        args (argparse.ArgumentParser): Global arguments for the training run.
    """
    # Load configurations file
    conf_file = Path("./config") / args.config
    with open(conf_file) as f:
        config = yaml.safe_load(f)

    set_float32_matmul_precision(config['matmul_32_precision'])

    # Set up data loaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config)
    num_classes = train_dataloader.dataset.dataset.num_classes

    # Initialize model
    module = BaseModule(config, num_classes)

    # Set up MLFlow Logger
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment(config["mlflow_experiment_name"])
    mlflow.pytorch.autolog()
    mlflow.start_run()

    mlflow.log_dict(config, 'config_file.txt')

    mlf_logger = MLFlowLogger(
        experiment_name=mlflow.get_experiment(
            mlflow.active_run().info.experiment_id).name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=mlflow.active_run().info.run_id,
    )

    # Set up Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=config['model_name'],
        save_top_k=1,
        mode='min'
    )

    # Set up Early Stopping
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min")

    # Set up trainer options for debugger
    trainer_kwargs = {}
    if config['debug']:
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        trainer_kwargs.update({
            'fast_dev_run': 10
        })

        if 'profiler' in config:
            if config['profiler'] == 'pytorch':
                profiler = PyTorchProfiler(
                    dirpath='./profiler',
                    filename=f'{timestamp}_pyt_prof'
                )
            elif config['profiler'] == 'advanced':
                profiler = AdvancedProfiler(
                    dirpath='./profiler',
                    filename=f'{timestamp}_prof'
                )
            else:
                raise ValueError(
                    f"Profiler type {config['profiler']} is not available.")
            trainer_kwargs.update({'profiler': profiler})

    # Set up the Trainer
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        **trainer_kwargs
    )

    # Train the model
    trainer.fit(module, train_dataloader, val_dataloader)

    # Evaluate on test dataset
    trainer.test(module, test_dataloader)

    # Log your model's artifacts to MLflow
    mlflow.pytorch.log_model(module, "model")

    # End the MLflow experiment
    mlflow.end_run()


def parse_args():
    """
    Parse the config file used for declaring training configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
