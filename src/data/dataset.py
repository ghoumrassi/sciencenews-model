from pathlib import Path
from torch.utils.data import Dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from transformers import AutoTokenizer
import torch
import h5py
from .utils import convert_to_hdf5, convert_to_hdf5_clicks


DATA_DIR = Path('data')


class LocalMINDDataset(Dataset):
    def __init__(
        self,
        text_col='title',
        target_col='category',
        max_length=512,
        tokenizer_name='distilbert-base-uncased',
        allow_download=False
    ):
        super().__init__()
        self.encoding = 'utf-8'
        self.max_length = max_length
        self.text_col = text_col
        self.target_col = target_col
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.h5_file = self._get_h5_filepath()
        self.news_file = self._get_news_filepath()
        self.headers = (
            'news_id', 'category', 'subcategory', 'title', 'abstract',
            'url', 'title_entities', 'abstract_entities'
        )

        if not self.news_file.exists():
            if not allow_download:
                raise FileNotFoundError(
                    "'news.tsv' could not be found but redownloading is not "
                    "allowed.\nPlease either allow downloading with "
                    "`allow_download`: true or manually install the file.")
            # If the data file is not available, download it using Kaggle API
            api = KaggleApi()

            # Authenticate with Kaggle API
            api.authenticate()
            api.dataset_download_files(
                'arashnic/mind-news-dataset',
                path=(DATA_DIR / 'raw'),
                unzip=True
            )
        if not self.h5_file.exists():
            self._create_h5()

        self.num_samples = 0
        self.label_idx_map = {}
        self.num_classes = 0
        self._load_h5_shape()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Open the HDF5 file
        with h5py.File(self.h5_file, 'r') as h5:
            # Access the HDF5 dataset
            x = h5[self.text_col][idx]
            y = h5[self.target_col][idx]

        # Tokenize x using the Huggingface Transformers tokenizer
        encoded = self.tokenizer.encode_plus(
            x.decode("utf-8"),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        x = encoded['input_ids'].squeeze()
        mask = encoded['attention_mask'].squeeze()

        # Convert y from a string label to an integer index
        y = self.label_idx_map[y]

        return (x, mask), y

    def get_headers(self):
        return self.headers

    def _create_h5(self):
        convert_to_hdf5(
            (DATA_DIR / 'raw' / 'MINDlarge_train' / 'news.tsv'),
            self.h5_file,
            'local-mind-large',
            self.headers
        )

    def _load_h5_shape(self):
        # Get the total number of observations
        # Open the HDF5 file
        with h5py.File(self.h5_file, 'r') as h5:
            # Get the number of samples (rows) in the HDF5 dataset
            self.num_samples = h5[self.text_col].shape[0]
            # Create a mapping of target labels to integer indices
            self.label_idx_map = {
                label: i for i, label in enumerate(
                    set(v for v in h5[self.target_col])
                )
            }
        self.num_classes = len(self.label_idx_map)

    def _get_h5_filepath(self):
        return (DATA_DIR / 'processed' / 'news.hdf5')

    def _get_news_filepath(self):
        return (DATA_DIR / 'raw' / 'MINDlarge_train' / 'news.tsv')


class LocalMINDDatasetClicks(LocalMINDDataset):
    def __init__(self, **kwargs):
        self.clicks_file = (
            DATA_DIR / 'raw' / 'MINDlarge_train' / 'behaviors.tsv'
        )
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        # Open the HDF5 file
        with h5py.File(self.h5_file, 'r') as h5:
            # Access the HDF5 dataset
            x = h5[self.text_col][idx]
            y = h5[self.target_col][idx]

        # Tokenize x using the Huggingface Transformers tokenizer
        encoded = self.tokenizer.encode_plus(
            x.decode("utf-8"),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        x = encoded['input_ids'].squeeze()
        mask = encoded['attention_mask'].squeeze()

        return (x, mask), torch.tensor(y, dtype=torch.float)

    def _create_h5(self):
        convert_to_hdf5_clicks(
            (DATA_DIR / 'raw' / 'MINDlarge_train' / 'news.tsv'),
            self.h5_file,
            self.clicks_file,
            'local-mind-large',
            self.headers
        )

    def _load_h5_shape(self):
        # Get the total number of observations
        # Open the HDF5 file
        with h5py.File(self.h5_file, 'r') as h5:
            # Get the number of samples (rows) in the HDF5 dataset
            self.num_samples = h5[self.text_col].shape[0]

    def _get_h5_filepath(self):
        return (DATA_DIR / 'processed' / 'news_clicks.hdf5')

    def _get_news_filepath(self):
        return (DATA_DIR / 'raw' / 'MINDlarge_train' / 'news.tsv')


if __name__ == "__main__":
    import cProfile
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    d = LocalMINDDataset()

    # Create a data loader for your dataset
    loader = DataLoader(d, batch_size=32, shuffle=True, num_workers=8)

    # Define a function to profile the dataset iteration
    def profile_dataset_iteration():
        for batch in tqdm(loader):
            pass

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Call the function to profile the dataset iteration
    profile_dataset_iteration()

    # Stop profiling
    profiler.disable()

    # Print profiling results
    profiler.print_stats()
