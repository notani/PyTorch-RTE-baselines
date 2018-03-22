'''
Script to generate a TSV file of predictions on the test data.
'''

import os
import importlib
import random
import pickle
import torch

from models.cbow import CBOW
from models.classifier import Classifier
from util import logger
from util.data_processing import *
from util.evaluate import *
import util.parameters as params

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS['model_name']
logpath = os.path.join(FIXED_PARAMETERS['log_path'], modname) + '.log'
logger = logger.Logger(logpath)
gpu = torch.cuda.is_available() and FIXED_PARAMETERS['gpu']
if gpu:
    print('Use GPU')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings. 
logger.Log('FIXED_PARAMETERS\n %s' % FIXED_PARAMETERS)


if __name__ == '__main__':
    # Load data
    logger.Log('Loading data')
    if FIXED_PARAMETERS['path_input'] is None:
        logger.Log('--input is empty')
        exit(1)
    data = load_nli_data(FIXED_PARAMETERS['path_input'])
    dictpath = os.path.join(FIXED_PARAMETERS['log_path'], modname) + '.p'

    if not os.path.isfile(dictpath):
        logger.Log('No dictionary found!')
        exit(1)
    logger.Log('Loading dictionary from %s' % (dictpath))
    word_indices = pickle.load(open(dictpath, 'rb'))
    logger.Log('Padding and indexifying sentences')
    sentences_to_padded_index_sequences(word_indices, [data])

    loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS['embedding_data_path'],
                                           word_indices)

    classifier = Classifier(CBOW, embeddings=loaded_embeddings,
                            params=FIXED_PARAMETERS, gpu=gpu, logger=logger)
    classifier.restore(os.path.join(FIXED_PARAMETERS['ckpt_path'], modname + '.ckpt_best'))

    logger.Log('Creating TSV of predicitons on matched test set: %s' %(modname+'_predictions.csv'))
    make_predictions(classifier.classify, data, modname)
