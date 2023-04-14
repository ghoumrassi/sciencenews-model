from operator import attrgetter
from torch.utils.data import DataLoader, random_split
from . import dataset as dataset_module


def get_dataloaders(config):
    dataset_fn = attrgetter(config['dataset']['name'])(dataset_module)
    dataset_args = config['dataset']['args']\
        if config['dataset']['args']\
        else {}
    dataset = dataset_fn(**dataset_args)

    # Split dataset into train and validation sets
    train_size = int(len(dataset) * config['train_split'])
    test_size = int(len(dataset) * config['test_split'])
    val_size = len(dataset) - train_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        num_workers=config['num_workers'], shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        num_workers=config['num_workers'], shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        num_workers=config['num_workers'], shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
