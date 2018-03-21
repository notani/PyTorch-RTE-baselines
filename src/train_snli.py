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

from util import logger
from util.data_processing import *
from util.evaluate import *
import util.parameters as params

FIXED_PARAMETERS = params.load_parameters()
gpu = torch.cuda.is_available() and FIXED_PARAMETERS['gpu']
print('Use GPU')
modname = FIXED_PARAMETERS['model_name']
logpath = os.path.join(FIXED_PARAMETERS['log_path'], modname) + '.log'
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS['model_type']

module = importlib.import_module('.'.join(['models', model])) 
MyModel = getattr(module, 'CBOW')

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

logger.Log('Loading embeddings')
loaded_embeddings = loadEmbedding_rand(FIXED_PARAMETERS['embedding_data_path'],
                                       word_indices)

class modelClassifier:
    def __init__(self, seq_length):
        ## Define hyperparameters
        self.learning_rate =  FIXED_PARAMETERS['learning_rate']
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS['word_embedding_dim']
        self.dim = FIXED_PARAMETERS['hidden_embedding_dim']
        self.batch_size = FIXED_PARAMETERS['batch_size']
        self.emb_train = FIXED_PARAMETERS['emb_train']
        self.keep_rate = FIXED_PARAMETERS['keep_rate']
        self.sequence_length = FIXED_PARAMETERS['seq_length'] 
        self.alpha = FIXED_PARAMETERS['alpha']

        logger.Log('Building model from %s.py' %(model))
        self.model = MyModel(hidden_dim=self.dim,
                             embeddings=loaded_embeddings,
                             keep_rate_ph=self.keep_rate)
        if gpu:
            self.model = self.model.cuda()

        # Boolean stating that training has not been completed, 
        self.completed = False 


    def get_minibatch(self, dataset, start_index, end_index):
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence']
                                     for i in indices])
        premise_vectors = torch.from_numpy(premise_vectors).long()
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence']
                                        for i in indices])
        hypothesis_vectors = torch.from_numpy(hypothesis_vectors).long()
        genres = [dataset[i]['genre'] for i in indices]
        labels = torch.LongTensor([dataset[i]['label'] for i in indices])
        if gpu:
            premise_vectors = premise_vectors.cuda()
            hypothesis_vectors = hypothesis_vectors.cuda()
            labels = labels.cuda()

        return premise_vectors, hypothesis_vectors, labels, genres


    def train(self, train_mnli, train_snli, dev_mat, dev_mismat, dev_snli):
        self.step = 1
        self.epoch = 0
        self.best_dev_snli = 0.
        self.best_strain_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0

        # Restore most recent checkpoint if it exists. 
        # Also restore values for best dev-set accuracy and best training-set accuracy.
        ckpt_file = os.path.join(FIXED_PARAMETERS['ckpt_path'], modname) + '.ckpt'
        if os.path.isfile(ckpt_file + '.meta'):
            if os.path.isfile(ckpt_file + '_best.meta'):
                self.restore(path=ckpt_file + '_best')
                # best_dev_mat, dev_cost_mat = evaluate_classifier(
                #     self.classify, dev_mat, self.batch_size)
                # best_dev_mismat, dev_cost_mismat = evaluate_classifier(
                #     self.classify, dev_mismat, self.batch_size)
                self.best_dev_snli, dev_cost_snli = evaluate_classifier(
                    self.classify, dev_snli, self.batch_size)
                self.best_strain_acc, strain_cost = evaluate_classifier(
                    self.classify, train_snli[0:5000], self.batch_size)
                # logger.Log('Restored best matched-dev acc: %f\n'
                #            ' Restored best mismatched-dev acc: %f\n'
                #            ' Restored best SNLI-dev acc: %f\n'
                #            ' Restored best SNLI train acc: %f'
                #            %(best_dev_mat, best_dev_mismat, self.best_dev_snli,
                #              self.best_strain_acc))
                logger.Log('Restored best SNLI-dev acc: %f\n'
                           ' Restored best SNLI train acc: %f'
                           %(self.best_dev_snli, self.best_strain_acc))

            self.restore(path=ckpt_file)
            logger.Log('Model restored from file: %s' % ckpt_file)

        training_data = train_snli

        ### Training cycle
        logger.Log('Training...')

        # Perform gradient descent with Adam
        loss_fn = nn.CrossEntropyLoss()
        if gpu:
            loss_fn = loss_fn.cuda()
        optimizer = optim.Adam(self.model.parameters(),
                               self.learning_rate,
                               betas=(0.9, 0.999))

        while True:
            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)
            
            # Loop over all batches in epoch
            for i in trange(total_batch):
                # Assemble a minibatch of the next B examples
                batch = self.get_minibatch(
                    training_data, self.batch_size * i, self.batch_size * (i + 1))
                minibatch_premise_vectors = batch[0]
                minibatch_hypothesis_vectors = batch[1]
                minibatch_labels = batch[2]
                minibatch_genres = batch[3]

                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function
                logit = self.model.forward(Variable(minibatch_premise_vectors),
                                           Variable(minibatch_hypothesis_vectors))
                loss = loss_fn(logit, Variable(minibatch_labels))
                loss.backward()
                optimizer.step()

                # # Print accuracy every (self.display_step_freq) steps
                # if self.step % self.display_step_freq == 0:
                #     # dev_acc_mat, dev_cost_mat = evaluate_classifier(
                #     #     self.classify, dev_mat, self.batch_size)
                #     # dev_acc_mismat, dev_cost_mismat = evaluate_classifier(
                #     #     self.classify, dev_mismat, self.batch_size)
                #     dev_acc_snli, dev_cost_snli = evaluate_classifier(
                #         self.classify, dev_snli, self.batch_size)
                #     strain_acc, strain_cost = evaluate_classifier(
                #         self.classify, train_snli[0:5000], self.batch_size)

                #     logger.Log('Step: %i\t Dev-SNLI acc: %f\t SNLI train acc: %f' %(self.step, dev_acc_snli, strain_acc))
                #     # logger.Log('Step: %i\t Dev-matched acc: %f\t Dev-mismatched acc: %f\t Dev-SNLI acc: %f\t SNLI train acc: %f' %(self.step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, strain_acc))
                #     logger.Log('Step: %i\t Dev-SNLI cost: %f\t SNLI train cost: %f' %(self.step, dev_cost_snli, strain_cost))
                #     # logger.Log('Step: %i\t Dev-matched cost: %f\t Dev-mismatched cost: %f\t Dev-SNLI cost: %f\t SNLI train cost: %f' %(self.step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, strain_cost))

                if self.step % 1000 == 0:
                    self.save(ckpt_file, quiet=True)
                    dev_acc_snli, _ = evaluate_classifier(
                        self.classify, dev_snli, self.batch_size)
                    strain_acc, _ = evaluate_classifier(
                        self.classify, train_snli[0:5000], self.batch_size)
                    if self.best_dev_snli < dev_acc_snli:
                        self.save(ckpt_file + '_best', quiet=True)
                        self.best_dev_snli = dev_acc_snli
                        self.best_strain_acc = strain_acc
                        self.best_step = self.step
                        logger.Log('Checkpointing with new best SNLI-dev accuracy: %f' %(self.best_dev_snli))

                self.step += 1

                # Compute average loss
                avg_cost += loss / total_batch

            # Display learning progress
            logger.Log('Epoch: %i\t Avg. Cost: %f' %(self.epoch+1, avg_cost))
            dev_acc_snli, dev_cost_snli = evaluate_classifier(
                self.classify, dev_snli, self.batch_size)
            strain_acc, strain_cost = evaluate_classifier(
                self.classify, train_snli[0:5000], self.batch_size)
            logger.Log('Step: %i\t Dev-SNLI acc: %f\t SNLI train acc: %f' %(self.step, dev_acc_snli, strain_acc))
            logger.Log('Step: %i\t Dev-SNLI cost: %f\t SNLI train cost: %f' %(self.step, dev_cost_snli, strain_cost))

            self.epoch += 1 
            self.last_train_acc[(self.epoch % 5) - 1] = strain_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1) 

            if (progress < 0.1) or (self.step > self.best_step + 30000):
                logger.Log('Best snli-dev accuracy\t: %s' %(self.best_dev_snli))
                logger.Log('Best snli-train accuracy\t: %s' %(self.best_strain_acc))
                self.completed = True
                break

    def save(self, path, quiet=False):
        torch.save(self.model.state_dict(), path)
        if not quiet:
            logger.Log('Model saved to file: %s' % path)

    def restore(self, path=None, best=True):
        if path is None:
            if best:
                path = os.path.join(FIXED_PARAMETERS['ckpt_path'], modname) + '.ckpt_best'
            else:
                path = os.path.join(FIXED_PARAMETERS['ckpt_path'], modname) + '.ckpt'

        self.model.load_state_dict(path)
        logger.Log('Model restored from file: %s' % path)

    def classify(self, examples):
        # This classifies a list of examples
        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(3)
        genres = []
        loss_fn = nn.CrossEntropyLoss(size_average=False)
        if gpu:
            loss_fn = loss_fn.cuda()
        loss = 0.0
        for i in range(total_batch):
            batch = self.get_minibatch(
                examples, self.batch_size * i, self.batch_size * (i + 1))
            minibatch_premise_vectors = batch[0]
            minibatch_hypothesis_vectors = batch[1]
            minibatch_labels = batch[2]
            minibatch_genres = batch[3]
            genres += minibatch_genres

            logit = self.model.forward(Variable(minibatch_premise_vectors),
                                       Variable(minibatch_hypothesis_vectors))
            loss += float(loss_fn(logit, Variable(minibatch_labels)))

            if gpu:
                logit = logit.cpu()
            logit = logit.data.numpy()
            logits = np.vstack([logits, logit])

        return genres, np.argmax(logits[1:], axis=1), loss


# Entry point
if __name__ == '__main__':
    random.seed(42)

    classifier = modelClassifier(FIXED_PARAMETERS['seq_length'])

    test = params.train_or_test()

    # Use dev-sets for testing
    test_matched = dev_matched
    test_mismatched = dev_mismatched

    if test == False:
        classifier.train(training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli)
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
