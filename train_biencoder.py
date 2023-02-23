"""
Example script for training a single biencoder model
"""

import pandas as pd
from sentence_transformers.losses import SiameseDistanceMetric

from modified_sbert.train import train_biencoder


def extract_raw_data(dataset_path, sep = ','):

    """
    Currently this is an example of how you can import data from a csv file.
    It can be amended to deal with other types of data.
    Labels should be 1 if same, 0 if different
    """

    raw_data = pd.read_csv(dataset_path, sep=sep, encoding='utf-8')

    sentence_1_list = [str(i) for i in list(raw_data["sentence_1"])]
    sentence_2_list = [str(i) for i in list(raw_data["sentence_2"])]
    labels = list(raw_data["labels"])

    assert len(sentence_1_list) == len(sentence_2_list) == len(labels)

    return {'sentence_1': sentence_1_list, 'sentence_2': sentence_2_list, "labels": labels}


if __name__ == '__main__':

    DATA_PATH = ''  # Update
    NAME = 'test'   # Update

    train_biencoder(
        train_data=extract_raw_data(f'{DATA_PATH}/train_set.csv', sep='\t'),
        dev_data=extract_raw_data(f'{DATA_PATH}/dev_set.csv', sep='\t'),
        base_model='sentence-transformers/all-mpnet-base-v2',
        add_pooling_layer=False,    # Switching to true if you use a non-sbert base model
        train_batch_size=16,
        num_epochs=10,
        warm_up_perc=0.5,
        loss_fn='contrastive',      # Options: 'supcon', 'cosine', 'triplet', 'mnrl', 'contrastive'
        loss_params={'distance_metric': SiameseDistanceMetric.COSINE_DISTANCE, 'margin': 0.5},
        model_save_path=f'/{NAME}',    # Update
        wandb_names={'project': "", "id": '', "run": NAME}   # Update
    )
