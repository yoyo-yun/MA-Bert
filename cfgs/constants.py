import os
from maa_datasets.dataset import IMDB, YELP_13, YELP_14

def ensureDirs(*dir_paths):
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

PRE_TRAINED_VECTOR_PATH = 'pretrained_Vectors'
PRE_TRAINED_BERT_PATH = 'pretrained_model'
SAVED_MODEL_PATH = 'saved_models'
DATASET_PATH = 'corpus'

ensureDirs(PRE_TRAINED_BERT_PATH, PRE_TRAINED_VECTOR_PATH, SAVED_MODEL_PATH)

DATASET_PATH_MAP = {
    "imdb": os.path.join(DATASET_PATH, 'imdb'),
    "yelp_13": os.path.join(DATASET_PATH, 'yelp_13'),
    "yelp_14": os.path.join(DATASET_PATH, 'yelp_14')
}

LABLES = {
    "imdb": 10,
    "yelp_13": 5,
    "yelp_14": 5
}

DATASET_MAP = {
    "imdb": IMDB,
    "yelp_13": YELP_13,
    "yelp_14": YELP_14,
}
