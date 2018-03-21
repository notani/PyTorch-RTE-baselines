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
    logger.Log('Loading dictionary from %s' % (dictpath))
    word_indices = pickle.load(open(dictpath, 'rb'))
    logger.Log('Padding and indexifying sentences')
    datasets = [training_mnli, training_snli, dev_matched, dev_mismatched,
                dev_snli, test_snli, test_matched, test_mismatched]
    sentences_to_padded_index_sequences(word_indices, datasets)

# Entry point
if __name__ == '__main__':
    random.seed(42)

    logger.Log('Loading embeddings')
    loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS['embedding_data_path'],
                                           word_indices)

    classifier = Classifier(CBOW, embeddings=loaded_embeddings,
                            params=FIXED_PARAMETERS, gpu=gpu, logger=logger)

    test = params.train_or_test()

    # Use dev-sets for testing
    test_matched = dev_matched
    test_mismatched = dev_mismatched

    if test == False:
        ckpt_file = os.path.join(FIXED_PARAMETERS['ckpt_path'], modname) + '.ckpt'
        classifier.train(training_snli, dev_snli, ckpt_file=ckpt_file)
        logger.Log('Acc on matched multiNLI dev-set: %s' %(evaluate_classifier(classifier.classify, test_matched, FIXED_PARAMETERS['batch_size']))[0])
        logger.Log('Acc on mismatched multiNLI dev-set: %s' %(evaluate_classifier(classifier.classify, test_mismatched, FIXED_PARAMETERS['batch_size']))[0])
        logger.Log('Acc on SNLI test-set: %s' %(evaluate_classifier(classifier.classify, test_snli, FIXED_PARAMETERS['batch_size']))[0])
    else:
        results = evaluate_final(classifier.restore, classifier.classify, [test_matched, test_mismatched, test_snli], FIXED_PARAMETERS['batch_size'])
        logger.Log('Acc on multiNLI matched dev-set: %s' %(results[0]))
        logger.Log('Acc on multiNLI mismatched dev-set: %s' %(results[1]))
        logger.Log('Acc on SNLI test set: %s' %(results[2]))

        # Results by genre,
        logger.Log('Acc on matched genre dev-sets: %s' %(evaluate_classifier_genre(classifier.classify, test_matched, FIXED_PARAMETERS['batch_size'])[0]))
        logger.Log('Acc on mismatched genres dev-sets: %s' %(evaluate_classifier_genre(classifier.classify, test_mismatched, FIXED_PARAMETERS['batch_size'])[0]))
