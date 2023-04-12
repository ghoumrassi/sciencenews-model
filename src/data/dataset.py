import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from transformers import AutoTokenizer


DATA_DIR = Path('data/raw')


class LocalMINDDataset(Dataset):
    def __init__(
        self,
        text_col='Title',
        target_col='Category',
        max_length=512,
        tokenizer_name='distilbert-base-uncased'
    ):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        data_pth = (DATA_DIR / 'MINDsmall_train')
        if not data_pth.exists():
            # If the data file is not available, download it using Kaggle API
            api = KaggleApi()

            # Authenticate with Kaggle API
            api.authenticate()
            api.dataset_download_files(
                'arashnic/mind-news-dataset', path='data/raw', unzip=True
            )
        news_file = data_pth / 'news.tsv'
        # Load the data into Numpy
        # Load TSV file into a pandas DataFrame
        df = pd.read_csv(news_file, sep='\t')
        # Convert DataFrame to numpy array
        self.data = df.to_numpy()

        # Read the column headers
        self.headers = (
            'news_id', 'category', 'subcategory', 'title', 'abstract',
            'url', 'title_entities', 'abstract_entities'
        )

        # Get the index of the text and target columns
        self.text_col = self.headers.index(text_col)
        self.target_col = self.headers.index(target_col)

        # We only need the text and the target
        self.data = self.data[:, (self.text_col, self.target_col)]

        # Create a mapping of target labels to integer indices
        self.label_idx_map = {
            label: i for i, label in enumerate(
                np.unique(self.data[:, self.target_col])
            )
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, 0]
        y = self.data[idx, 1]

        # Tokenize x using the Huggingface Transformers tokenizer
        encoded = self.tokenizer.encode_plus(
            x,
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
