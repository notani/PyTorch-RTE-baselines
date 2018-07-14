'''
Script to generate a TSV file of predictions on the test data.
'''

import os
import importlib
import random
import pickle
import torch


from models import cbow_sg
from models import cbow
from models.classifier import Classifier
from util import logger
from util.data_processing import *
from util.evaluate import *
import util.parameters as params

args = params.load_parameters()
modname = args['model_name']
logpath = os.path.join(args['log_path'], modname) + '.log'
logger = logger.Logger(logpath)
gpu = torch.cuda.is_available() and args['gpu']
if gpu:
    print('Use GPU')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings. 
logger.Log('args\n %s' % args)


if __name__ == '__main__':
    # Load data
    logger.Log('Loading data')
    if args['path_input'] is None:
        logger.Log('--input is empty')
        exit(1)
    data = load_nli_data(args['path_input'])
    dictpath = os.path.join(args['log_path'], modname) + '.p'

    if not os.path.isfile(dictpath):
        logger.Log('No dictionary found!')
        exit(1)
    logger.Log('Loading dictionary from %s' % (dictpath))
    word_indices = pickle.load(open(dictpath, 'rb'))
    logger.Log('Padding and indexifying sentences')
    sentences_to_padded_index_sequences(word_indices, [data])

    loaded_embeddings = loadEmbedding_rand(args['embedding_data_path'],
                                           word_indices)
    if args['flag_sg']:
        model = cbow_sg.CBOWSG
        Classifier = cbow_sg.Classifier
    else:
        model = cbow.CBOW
        Classifier = cbow.Classifier
    clf = Classifier(model, embeddings=loaded_embeddings,
                     params=args, gpu=gpu, logger=logger)
    clf.restore(os.path.join(args['ckpt_path'], modname) + '.ckpt_best', gpu=gpu)

    logger.Log('Creating TSV of predicitons on matched test set: %s' %(modname+'_predictions.csv'))
    make_predictions(clf.classify, data, modname)
