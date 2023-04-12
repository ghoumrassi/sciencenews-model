from operator import attrgetter
import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
from models import model as models_module


class BaseModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_inst = attrgetter(config['model']['name'])(models_module)
        model_args = config['model']['args']\
            if 'args' in config['model'] else {}
        self.model = model_inst(**model_args)

        criterion_inst = attrgetter(config['loss']['name'])(nn)
        crit_args = config['loss']['args'] if 'args' in config['loss'] else {}
        self.criterion = criterion_inst(**crit_args)

    def forward(self, inputs):
        return self.model(*inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config['lr'])
        return optimizer
