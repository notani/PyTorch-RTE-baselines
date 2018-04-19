'''
Training script to train a model on only SNLI data. MultiNLI data is loaded into the embeddings enabling us to test the model on MultiNLI data.
'''

from torch import nn
from torch import optim
from torch.autograd import Variable
from tqdm import trange
import importlib
import os
import random
import torch
import numpy as np

from models.cbow import CBOW
from models.classifier import Classifier
from util import logger
from util.data_processing import *
from util.evaluate import *
import util.parameters as params

FIXED_PARAMETERS = params.load_parameters()
gpu = torch.cuda.is_available() and FIXED_PARAMETERS['gpu']
if gpu:
    print('Use GPU')
modname = FIXED_PARAMETERS['model_name']
logpath = os.path.join(FIXED_PARAMETERS['log_path'], modname) + '.log'
logger = logger.Logger(logpath)

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings. 
logger.Log('FIXED_PARAMETERS\n %s' % FIXED_PARAMETERS)

######################### LOAD DATA #############################

logger.Log('Loading data')
training_snli = load_nli_data(FIXED_PARAMETERS['training_snli'], snli=True)
dev_snli = load_nli_data(FIXED_PARAMETERS['dev_snli'], snli=True)
test_snli = load_nli_data(FIXED_PARAMETERS['test_snli'], snli=True)

training_mnli = load_nli_data(FIXED_PARAMETERS['training_mnli'])
dev_matched = load_nli_data(FIXED_PARAMETERS['dev_matched'])
dev_mismatched = load_nli_data(FIXED_PARAMETERS['dev_mismatched'])
test_matched = load_nli_data(FIXED_PARAMETERS['test_matched'])
test_mismatched = load_nli_data(FIXED_PARAMETERS['test_mismatched'])

if 'temp.jsonl' in FIXED_PARAMETERS['test_matched']:
    # Removing temporary empty file that was created in parameters.py
    os.remove(FIXED_PARAMETERS['test_matched'])
    logger.Log('Created and removed empty file called temp.jsonl since test set is not available.')

dictpath = os.path.join(FIXED_PARAMETERS['log_path'], modname) + '.p'

if not os.path.isfile(dictpath): 
    logger.Log('Building dictionary')
    word_indices = build_dictionary([training_snli])
    logger.Log('Padding and encoding sentences')
    datasets = [training_snli, training_mnli, dev_matched, dev_mismatched,
                dev_snli, test_snli, test_matched, test_mismatched]
    sentences_to_padded_index_sequences(word_indices, datasets)
    pickle.dump(word_indices, open(dictpath, 'wb'))
else:
    logger.Log('Loading dictionary from ' + dictpath)
    word_indices = pickle.load(open(dictpath, 'rb'))
    logger.Log('Padding and indexifying sentences')
    datasets = [training_mnli, training_snli, dev_matched, dev_mismatched,
                dev_snli, test_snli, test_matched, test_mismatched]
    sentences_to_padded_index_sequences(word_indices, datasets)

# Entry point
if __name__ == '__main__':
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    logger.Log('Loading embeddings')
    loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS['embedding_data_path'],
                                           word_indices)

    clf = Classifier(CBOW, embeddings=loaded_embeddings,
                     params=FIXED_PARAMETERS, gpu=gpu, logger=logger)

    test = params.train_or_test()

    # Use dev-sets for testing
    test_matched = dev_matched
    test_mismatched = dev_mismatched

    batch_size = FIXED_PARAMETERS['batch_size']
    if test == False:
        ckpt_file = os.path.join(FIXED_PARAMETERS['ckpt_path'], modname) + '.ckpt'
        clf.train(training_snli, dev_snli, ckpt_file=ckpt_file)
        logger.Log('Acc on SNLI test-set:\t{:.4f}'.format(
            evaluate_classifier(clf.classify, test_snli, batch_size)[0]))
        logger.Log('Acc on matched multiNLI dev-set:\t{:.4f}'.format(
            evaluate_classifier(clf.classify, test_matched, batch_size)[0]))
        logger.Log('Acc on mismatched multiNLI dev-set:\t{:.4f}'.format(
            evaluate_classifier(clf.classify, test_mismatched, batch_size)[0]))
    else:
        clf.restore(os.path.join(FIXED_PARAMETERS['ckpt_path'], modname) + '.ckpt_best')
        results, _ = evaluate_final(
            clf.classify, [test_matched, test_mismatched, test_snli], batch_size)
        logger.Log('Acc on SNLI test set: {:.4f}'.format(results[2]))
        logger.Log('Acc on multiNLI matched dev-set\t: {:.4f}'.format(results[0]))
        logger.Log('Acc on multiNLI mismatched dev-set: {:.4f}'.format(results[1]))

        # Results by genre,
        logger.Log('Acc on matched genre dev-sets: {}'.format(
            evaluate_classifier_genre(clf.classify, test_matched, batch_size)[0]))
        logger.Log('Acc on mismatched genres dev-sets: {}'.format(
            evaluate_classifier_genre(clf.classify, test_mismatched, batch_size)[0]))
