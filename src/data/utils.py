from collections import defaultdict
import csv
import h5py
import pandas as pd
from tqdm import tqdm


def convert_to_hdf5(
        csv_file, hdf5_file, name, headers, encoding='utf-8', chunksize=10000):
    # Define the structure of your CSV file
    num_columns = len(headers)  # Set the number of columns in your CSV file
    column_dtypes = [
        h5py.string_dtype(encoding=encoding, length=None)
    ] * num_columns

    # Create an empty HDF5 file with the correct structure
    with h5py.File(hdf5_file, 'w') as f:
        # Create an empty resizable dataset for each column
        for name, dtype in zip(headers, column_dtypes):
            f.create_dataset(name, (0,), maxshape=(None,), dtype=dtype)

    # Read the CSV file in chunks and append the data to the HDF5 file
    with h5py.File(hdf5_file, 'a') as f:
        for chunk in pd.read_csv(
                csv_file, chunksize=chunksize,
                names=headers, delimiter='\t'):

            chunk.fillna('', inplace=True)
            # Get the current size of the HDF5 datasets
            current_size = f[headers[0]].shape[0]

            # Calculate the new size after appending the chunk
            new_size = current_size + chunk.shape[0]

            # Resize the HDF5 datasets to accommodate the new data
            for name in headers:
                f[name].resize(new_size, axis=0)

            # Append the chunk data to the HDF5 datasets
            for name in headers:
                f[name][current_size:new_size] = chunk[name].to_numpy()


def convert_to_hdf5_clicks(
        csv_file, hdf5_file, clicks_file, name, headers, encoding='utf-8',
        chunksize=10000):

    # Get the clicks and CTR for each article
    article_stats = defaultdict(
        lambda: {'impressions': 0, 'clicks': 0, 'ctr': 0})

    with open(clicks_file, mode='r', newline='') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for i, row in tqdm(enumerate(tsv_reader)):
            impressions = row[4].split(' ')

            for impression in impressions:
                article_id, click = impression.split('-')
                article_stats[article_id]['impressions'] += 1

                if click == '1':
                    article_stats[article_id]['clicks'] += 1

    # Calculate click-through rates
    for article_id, stats in article_stats.items():
        stats['ctr'] = stats['clicks'] / stats['impressions']

    # Convert defaultdict to list of dictionaries
    click_stats_df = pd.DataFrame(
        [{'news_id': k, **v} for k, v in article_stats.items()]
    )

    # Define the structure of your CSV file
    num_columns = len(headers)  # Set the number of columns in your CSV file
    column_dtypes = [
        h5py.string_dtype(encoding=encoding, length=None)
    ] * num_columns

    # Add in additional columns
    headers_plus = list(headers) + ['impressions', 'clicks', 'ctr']
    column_dtypes = column_dtypes + [int, int, float]

    # Create an empty HDF5 file with the correct structure
    with h5py.File(hdf5_file, 'w') as f:
        # Create an empty resizable dataset for each column
        for name, dtype in zip(headers_plus, column_dtypes):
            f.create_dataset(name, (0,), maxshape=(None,), dtype=dtype)

    # Read the CSV file in chunks and append the data to the HDF5 file
    with h5py.File(hdf5_file, 'a') as f:
        for chunk in pd.read_csv(
                csv_file, chunksize=chunksize,
                names=headers, delimiter='\t'):

            chunk.fillna('', inplace=True)

            chunk = pd.merge(chunk, click_stats_df, on='news_id', how='left')
            chunk.fillna(0, inplace=True)

            chunk = chunk[chunk['impressions'] > 1]

            # Get the current size of the HDF5 datasets
            current_size = f[headers_plus[0]].shape[0]

            # Calculate the new size after appending the chunk
            new_size = current_size + chunk.shape[0]

            # Resize the HDF5 datasets to accommodate the new data
            for name in headers_plus:
                f[name].resize(new_size, axis=0)

            # Append the chunk data to the HDF5 datasets
            for name in headers_plus:
                f[name][current_size:new_size] = chunk[name].to_numpy()
