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

from models import cbow_sg
from models import cbow
# from models.cbow import CBOW
# from models.classifier import Classifier
from util import logger
from util.data_processing import *
from util.evaluate import *
import util.parameters as params

# Entry point
if __name__ == '__main__':
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    args = params.load_parameters()
    gpu = torch.cuda.is_available() and args['gpu']
    if gpu:
        print('Use GPU')
    modname = args['model_name']
    logpath = os.path.join(args['log_path'], modname) + '.log'
    logger = logger.Logger(logpath)

    # Logging parameter settings at each launch of training script
    # This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings.
    logger.Log('args\n %s' % args)

    # Data loading
    logger.Log('Loading data')
    training_snli = load_nli_data(args['training_snli'])
    dev_snli = load_nli_data(args['dev_snli'])
    test_snli = load_nli_data(args['test_snli'])

    if 'temp.jsonl' in args['test_matched']:
        # Removing temporary empty file that was created in parameters.py
        os.remove(args['test_matched'])
        logger.Log('Created and removed empty file called temp.jsonl since test set is not available.')

    dictpath = os.path.join(args['log_path'], modname) + '.p'

    if not os.path.isfile(dictpath):
        logger.Log('Building dictionary')
        word_indices = build_dictionary([training_snli])
        logger.Log('Padding and encoding sentences')
        datasets = [training_snli, dev_snli, test_snli]
        sentences_to_padded_index_sequences(word_indices, datasets)
        pickle.dump(word_indices, open(dictpath, 'wb'))
    else:
        logger.Log('Loading dictionary from ' + dictpath)
        word_indices = pickle.load(open(dictpath, 'rb'))
        logger.Log('Padding and indexifying sentences')
        datasets = [training_snli, dev_snli, test_snli]
        sentences_to_padded_index_sequences(word_indices, datasets)

    logger.Log('Loading embeddings')
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

    test = params.train_or_test()

    batch_size = args['batch_size']
    if test:  # test only
        clf.restore(os.path.join(args['ckpt_path'], modname) + '.ckpt_best')
        results, _ = evaluate_final(
            clf.classify, [test_snli], batch_size)
        logger.Log('Acc on Test set: {:.4f}'.format(results[0]))

        # Results by genre,
        logger.Log('Acc by genre: {}'.format(
            evaluate_classifier_genre(clf.classify, test_snli, batch_size)[0]))
    else:
        ckpt_file = os.path.join(args['ckpt_path'], modname) + '.ckpt'
        clf.train(training_snli, dev_snli, ckpt_file=ckpt_file)
        logger.Log('Acc on SNLI test-set:\t{:.4f}'.format(
            evaluate_classifier(clf.classify, test_snli, batch_size)[0]))
