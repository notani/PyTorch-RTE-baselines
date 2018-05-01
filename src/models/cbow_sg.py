from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm
from tqdm import trange
import numpy as np
import os
import random
import torch

from util.evaluate import evaluate_classifier


class CBOWSG(nn.Module):
    def __init__(self, hidden_dim, embeddings, keep_rate):
        super(CBOWSG, self).__init__()
        self.__name__ = 'CBOW-SG'

        ## Define hyperparameters
        self.embedding_dim = embeddings.shape[1]
        self.dim = hidden_dim
        self.dropout_rate = 1.0 - keep_rate

        ## Define remaning parameters 
        self.E = nn.Embedding(embeddings.shape[0], self.embedding_dim, padding_idx=0)
        self.E.weight.data.copy_(torch.from_numpy(embeddings))
        self.W_0 = nn.Linear(self.embedding_dim * 6, self.dim)
        self.W_1 = nn.Linear(self.dim, self.dim)
        self.W_2 = nn.Linear(self.dim, self.dim)
        self.W_out = nn.Linear(self.dim, 3)
        self.W_enc_attr = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_enc_pred = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_enc_objt = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_enc = [self.W_enc_attr, self.W_enc_pred, self.W_enc_objt]
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.W_0.weight, std=0.1)
        nn.init.normal_(self.W_0.bias, std=0.1)
        nn.init.normal_(self.W_1.weight, std=0.1)
        nn.init.normal_(self.W_1.bias, std=0.1)
        nn.init.normal_(self.W_2.weight, std=0.1)
        nn.init.normal_(self.W_2.bias, std=0.1)
        nn.init.normal_(self.W_out.weight, std=0.1)
        nn.init.normal_(self.W_out.bias, std=0.1)
        for W in self.W_enc:
            nn.init.normal_(W.weight, std=0.1)

    def encode_sg(self, sg, seq, train=False):
        if sg.word_idx == -1:  # root
            word = Variable(torch.ones(seq[0].size()))
            if self.E.weight.is_cuda:
                word = word.cuda()
        else:
            word = seq[sg.word_idx]
            if train:
                word = F.dropout(word, p=self.dropout_rate)

        if sg.num_children == 0:
            return word

        children = []
        for child, rel_id in sg.children:
            h = self.encode_sg(child, seq, train=train)
            if rel_id == -1:  # top-level
                pass
            elif rel_id == 3:  # same
                pass
            else:
                h = self.W_enc[rel_id](h)
            children.append(h)
        agg = torch.cat(children, dim=0) # aggregate children
        if train:
            agg = F.dropout(agg, p=self.dropout_rate)
        agg = word * agg.mean(dim=0)
        return F.relu(agg)

    def forward(self, premise_x, hypothesis_x, premise_sg, hypothesis_sg, train=False):
        # Calculate representaitons by CBOW method
        emb_premise = self.E(premise_x)
        if train:
            emb_premise_drop = F.dropout(emb_premise, p=self.dropout_rate)
        else:
            emb_premise_drop = emb_premise

        emb_hypothesis = self.E(hypothesis_x)
        if train:
            emb_hypothesis_drop = F.dropout(emb_hypothesis, p=self.dropout_rate)
        else:
            emb_hypothesis_drop = emb_hypothesis

        ## Sentence representations
        premise_rep = emb_premise_drop.sum(dim=1)
        hypothesis_rep = emb_hypothesis_drop.sum(dim=1)

        ## Scene graph representation
        premise_sg = torch.cat([self.encode_sg(sg, seq, train=train).view((1, -1))
                                for sg, seq in zip(premise_sg, emb_premise)], dim=0)
        hypothesis_sg = torch.cat([self.encode_sg(sg, seq, train=train).view((1, -1))
                                   for sg, seq in zip(hypothesis_sg, emb_hypothesis)], dim=0)

        ## Combinations
        h_diff = premise_rep - hypothesis_rep
        h_mul = premise_rep * hypothesis_rep
        sg_diff = premise_sg - hypothesis_sg
        sg_mul = premise_sg * hypothesis_sg

        ## MLP
        mlp_input = torch.cat([premise_rep, hypothesis_rep, h_diff, h_mul, sg_diff, sg_mul], 1)
        h_1 = F.relu(self.W_0(mlp_input))
        h_2 = F.relu(self.W_1(h_1))
        h_3 = F.relu(self.W_2(h_2))
        if train:
            h_drop = F.dropout(h_3, p=self.dropout_rate)
        else:
            h_drop = h_3

        # Get prediction
        return self.W_out(h_drop)


class Classifier:
    """Wrapper of classification models."""
    def __init__(self, model, embeddings, params, gpu=False, logger=None):
        ## Define hyperparameters
        self.gpu = gpu
        self.logger = logger
        if logger is None:
            raise
        self.learning_rate =  params['learning_rate']
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = params['word_embedding_dim']
        self.dim = params['hidden_embedding_dim']
        self.batch_size = params['batch_size']
        self.emb_train = params['emb_train']
        self.keep_rate = params['keep_rate']
        self.alpha = params['alpha']

        self.logger.Log('Building model: ' + model.__name__)
        self.model = model(hidden_dim=self.dim,
                           embeddings=embeddings,
                           keep_rate=self.keep_rate)
        if self.gpu:
            self.model = self.model.cuda()

        # Boolean stating that training has not been completed, 
        self.completed = False 


    def get_minibatch(self, dataset, start_index, end_index):
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([dataset[i]['sentence1_index_sequence']
                                     for i in indices])
        premise_vectors = torch.from_numpy(premise_vectors).long()
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_index_sequence']
                                        for i in indices])
        hypothesis_vectors = torch.from_numpy(hypothesis_vectors).long()
        premise_sg = [dataset[i]['sentence1_sg_tree'] for i in indices]
        hypothesis_sg = [dataset[i]['sentence2_sg_tree'] for i in indices]
        try:
            genres = [dataset[i]['genre'] for i in indices]
        except KeyError:
            genres = [None for i in indices]
        labels = torch.LongTensor([dataset[i]['label'] for i in indices])
        if self.gpu:
            premise_vectors = premise_vectors.cuda()
            hypothesis_vectors = hypothesis_vectors.cuda()
            labels = labels.cuda()

        return premise_vectors, hypothesis_vectors, premise_sg, hypothesis_sg, labels, genres


    def train(self, train, dev, ckpt_file):
        self.step = 1
        self.epoch = 0
        self.best_dev = 0.
        self.best_strain_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0

        # Restore most recent checkpoint if it exists. 
        # Also restore values for best dev-set accuracy and best training-set accuracy.
        if os.path.isfile(ckpt_file + '.meta'):
            if os.path.isfile(ckpt_file + '_best.meta'):
                self.restore(path=ckpt_file + '_best')
                self.best_dev, dev_cost = evaluate_classifier(
                    self.classify, dev, self.batch_size)
                self.best_strain_acc, strain_cost = evaluate_classifier(
                    self.classify, train[0:5000], self.batch_size)
                self.logger.Log('Restored best dev acc: %f\n'
                                ' Restored best train acc: %f'
                                %(self.best_dev, self.best_strain_acc))

            self.restore(path=ckpt_file)
            self.logger.Log('Model restored from file: %s' % ckpt_file)

        training_data = train

        ### Training cycle
        self.logger.Log('Training...')

        # Perform gradient descent with Adam
        loss_fn = nn.CrossEntropyLoss()
        if self.gpu:
            loss_fn = loss_fn.cuda()
        optimizer = optim.Adam(self.model.parameters(), self.learning_rate)

        while True:
            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)

            # Loop over all batches in epoch
            for i in trange(total_batch):
                # Assemble a minibatch of the next B examples
                batch = self.get_minibatch(
                    training_data, self.batch_size * i, self.batch_size * (i + 1))
                batch_premise_vectors = batch[0]
                batch_hypothesis_vectors = batch[1]
                batch_premise_sg = batch[2]
                batch_hypothesis_sg = batch[3]
                batch_labels = batch[4]
                batch_genres = batch[5]

                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function
                optimizer.zero_grad()
                logit = self.model.forward(Variable(batch_premise_vectors),
                                           Variable(batch_hypothesis_vectors),
                                           batch_premise_sg, batch_hypothesis_sg,
                                           train=True)
                loss = loss_fn(logit, Variable(batch_labels))
                loss.backward()
                optimizer.step()

                if self.step % 1000 == 0:
                    self.save(ckpt_file, quiet=True)
                    dev_acc, _ = evaluate_classifier(
                        self.classify, dev, self.batch_size)
                    strain_acc, _ = evaluate_classifier(
                        self.classify, train[0:5000], self.batch_size)
                    if self.best_dev < dev_acc:
                        self.save(ckpt_file + '_best', quiet=True)
                        self.best_dev = dev_acc
                        self.best_strain_acc = strain_acc
                        self.best_step = self.step
                        self.logger.Log('Checkpointing with new best dev accuracy: %f' %(self.best_dev))

                self.step += 1

                # Compute average loss
                avg_cost += loss / total_batch

            # Display learning progress
            self.logger.Log('Epoch: %i\t Avg. Cost: %f' %(self.epoch+1, avg_cost))
            dev_acc, dev_cost = evaluate_classifier(
                self.classify, dev, self.batch_size)
            train_acc, train_cost = evaluate_classifier(
                self.classify, train[0:5000], self.batch_size)
            self.logger.Log('Step: %i\tAcc\tdev: %f\ttrain: %f' %(
                self.step, dev_acc, train_acc))
            self.logger.Log('Step: %i\tCost\tdev: %f\ttrain: %f' %(
                self.step, dev_cost, train_cost))

            self.epoch += 1 
            self.last_train_acc[(self.epoch % 5) - 1] = train_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_acc)/(5 * min(self.last_train_acc)) - 1) 

            if (progress < 0.1) or (self.step > self.best_step + 30000):
                self.logger.Log('Best dev accuracy\t: %s' %(self.best_dev))
                self.logger.Log('Best train accuracy\t: %s' %(self.best_strain_acc))
                self.completed = True
                break

    def save(self, path, quiet=False):
        """Save model parameters."""
        torch.save(self.model.state_dict(), path)
        if not quiet:
            self.logger.Log('Model saved to file: ' + path)

    def restore(self, path):
        """Restore model parameters."""
        self.model.load_state_dict(torch.load(path))
        self.logger.Log('Model restored from file: ' + path)

    def classify(self, examples, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # This classifies a list of examples
        total_batch = int(len(examples) / batch_size)
        logits = np.empty(3)
        genres = []
        loss_fn = nn.CrossEntropyLoss(size_average=False)
        if self.gpu:
            loss_fn = loss_fn.cuda()
        loss = 0.0
        for i in range(total_batch):
            batch = self.get_minibatch(
                examples, batch_size * i, batch_size * (i + 1))
            batch_premise_vectors = batch[0]
            batch_hypothesis_vectors = batch[1]
            batch_premise_sg = batch[2]
            batch_hypothesis_sg = batch[3]
            batch_labels = batch[4]
            batch_genres = batch[5]
            genres += batch_genres

            logit = self.model.forward(Variable(batch_premise_vectors),
                                       Variable(batch_hypothesis_vectors),
                                       batch_premise_sg, batch_hypothesis_sg,
                                       train=False)
            loss += float(loss_fn(logit, Variable(batch_labels)))

            if self.gpu:
                logit = logit.cpu()
            logit = logit.data.numpy()
            logits = np.vstack([logits, logit])

        if batch_size * (i + 1) < len(examples):
            batch = self.get_minibatch(
                examples, batch_size * (i + 1), len(examples))
            batch_premise_vectors = batch[0]
            batch_hypothesis_vectors = batch[1]
            batch_premise_sg = batch[2]
            batch_hypothesis_sg = batch[3]
            batch_labels = batch[4]
            batch_genres = batch[5]
            genres += batch_genres

            logit = self.model.forward(Variable(batch_premise_vectors),
                                       Variable(batch_hypothesis_vectors),
                                       batch_premise_sg, batch_hypothesis_sg,
                                       train=False)
            loss += float(loss_fn(logit, Variable(batch_labels)))

            if self.gpu:
                logit = logit.cpu()
            logit = logit.data.numpy()
            logits = np.vstack([logits, logit])

        return genres, np.argmax(logits[1:], axis=1), loss
