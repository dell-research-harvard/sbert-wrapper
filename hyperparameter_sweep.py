from sentence_transformers.losses import SiameseDistanceMetric
import wandb

from train_biencoder import extract_raw_data
from modified_sbert.train import train_biencoder


def train():

    run = wandb.init()

    DATA_PATH = '' # Update
    NAME = f'{wandb.config.batch_size}_{wandb.config.epochs}_{wandb.config.warm_up_perc}_{wandb.config.loss_margin}'

    train_biencoder(
        train_data=extract_raw_data(f'{DATA_PATH}/train_set.csv', sep='\t'),
        dev_data=extract_raw_data(f'{DATA_PATH}/dev_set.csv', sep='\t'),
        base_model='sentence-transformers/all-mpnet-base-v2',
        add_pooling_layer=False,
        train_batch_size=wandb.config.batch_size,
        num_epochs=wandb.config.epochs,
        warm_up_perc=wandb.config.warm_up_perc,
        loss_fn='contrastive',
        loss_params={'distance_metric': SiameseDistanceMetric.COSINE_DISTANCE, 'margin': wandb.config.loss_margin},
        model_save_path=f'{NAME}'   # Update
    )


if __name__ == '__main__':

    # Config hyperparameter sweep
    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',   
        'metric': {'goal': 'maximize', 'name': "Classification F1 Cosine-Similarity"},
        'early_terminate': {'type': 'hyperband', 'min_iter': 100},    
        'parameters': 
            {
                'loss_margin': {'min': 0.1, 'max': 0.9},          # Needs specifying for constrative and triplet loss
                'batch_size': {'values': [16, 32, 48, 64]},       # Too big for some models, but should just fail and move on 
                'epochs': {'min': 4, 'max': 10},                                                  
                'warm_up_perc': {'min': 0.0, 'max': 1.0}
            }
    }

    # Update project and entity
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='',  entity="")
    wandb.agent(sweep_id, project='', entity="", function=train, count=200)
