from operator import attrgetter
import pytorch_lightning as pl
from torch import nn, cat, optim

from models import model as models_module
from .utils import (
    make_eval_metrics_classification, make_eval_metrics_regression)


class BaseModule(pl.LightningModule):
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        model_inst = attrgetter(config['model']['name'])(models_module)
        model_args = config['model']['args']\
            if 'args' in config['model'] else {}
        self.model = model_inst(**model_args)

        criterion_inst = attrgetter(config['loss']['name'])(nn)
        crit_args = config['loss']['args'] if 'args' in config['loss'] else {}
        self.criterion = criterion_inst(**crit_args)

        self.optimizer_inst = attrgetter(config['optimizer']['name'])(optim)
        self.optimizer_args = config['optimizer']['args']\
            if 'args' in config['optimizer'] else {}

        self.scheduler_inst = attrgetter(
            config['scheduler']['name'])(optim.lr_scheduler)
        self.scheduler_args = config['scheduler']['args']\
            if 'args' in config['scheduler'] else {}

        self.valid_y_accum = []
        self.valid_logit_accum = []
        self.test_y_accum = []
        self.test_logit_accum = []

    def forward(self, inputs):
        return self.model(*inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.valid_y_accum.append(y)
        self.valid_logit_accum.append(logits)

    def on_validation_epoch_end(self):
        y = cat(self.valid_y_accum).to('cpu')
        logits = cat(self.valid_logit_accum).to('cpu')
        if self.config['task'] == 'classification':
            make_eval_metrics_classification(self, y, logits, 'validation')
        elif self.config['task'] == 'regression':
            make_eval_metrics_regression(self, y, logits, 'validation')
        self.valid_y_accum = []
        self.valid_logit_accum = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss, on_epoch=True, on_step=False)
        self.test_y_accum.append(y)
        self.test_logit_accum.append(logits)

    def on_test_epoch_end(self):
        y = cat(self.test_y_accum).to('cpu')
        logits = cat(self.test_logit_accum).to('cpu')
        if self.config['task'] == 'classification':
            make_eval_metrics_classification(self, y, logits, 'test')
        elif self.config['task'] == 'regression':
            make_eval_metrics_regression(self, y, logits, 'test')

    def configure_optimizers(self):
        optimizer = self.optimizer_inst(self.parameters(),
                                        **self.optimizer_args)
        if not self.scheduler_inst:
            return optimizer

        scheduler = self.scheduler_inst(optimizer,
                                        **self.scheduler_args)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]
