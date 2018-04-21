import collections
import json
import numpy as np
import pickle
import random
import re

from util.tree import seq2tree
import util.parameters as params
  
PARAMS = params.load_parameters()

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

def load_nli_data(path):
    """Load MultiNLI or SNLI data.
    If the 'snli' parameter is set to True, a genre label of snli will be assigned to the data."""
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example['gold_label'] not in LABEL_MAP:
                continue
            loaded_example['label'] = LABEL_MAP[loaded_example['gold_label']]
            if 'genre' not in loaded_example:
                loaded_example['genre'] = 'snli'
            data.append(loaded_example)
        random.shuffle(data)
    return data

def load_nli_data_genre(path, genre, snli=True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            if loaded_example["genre"] == genre:
                data.append(loaded_example)
        random.shuffle(data)
    return data

def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()

def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['sentence1_binary_parse']))
            word_counter.update(tokenize(example['sentence2_binary_parse']))
        
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices

def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    REL = {'_': -1, 'ATTR': 0, 'PRED': 1, 'OBJT': 2, 'same': 3}
    for i, dataset in enumerate(datasets):
        for example in dataset:
            for i in [1, 2]:
                sentence = 'sentence' + str(i)
                sg = 'sg' + str(i)
                example[sentence + '_index_sequence'] = np.zeros((PARAMS['seq_length']), dtype=np.int32)
                example[sentence + '_sg_tree'] = []

                token_sequence = tokenize(example[sentence + '_binary_parse'])
                padding = PARAMS['seq_length'] - len(token_sequence)

                for i in range(PARAMS['seq_length']):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                    else:
                        try:
                            index = word_indices[token_sequence[i]]
                        except KeyError:
                            index = word_indices[UNKNOWN]
                    example[sentence + '_index_sequence'][i] = index
                    if i >= len(example[sg]):
                        continue
                    elem = example[sg][i]
                    example[sentence + '_sg_tree'].append(
                        [elem[0], index, elem[2], REL[elem[3]]])
                example[sentence + '_sg_tree'] = seq2tree(example[sentence + '_sg_tree'])


def loadEmbedding_zeros(path, word_indices):
    """
    Load GloVe embeddings. Initializng OOV words to vector of zeros.
    """
    emb = np.zeros((len(word_indices), PARAMS["word_embedding_dim"]), dtype='float32')
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if PARAMS["embeddings_to_load"] != None:
                if i >= PARAMS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb


def loadEmbedding_rand(path, word_indices):
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    n = len(word_indices)
    m = PARAMS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:2, :] = np.zeros((1,m), dtype="float32")

    count_errors = 0
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if PARAMS["embeddings_to_load"] != None:
                if i >= PARAMS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                try:
                    emb[word_indices[s[0]], :] = np.asarray(s[1:])
                except ValueError:
                    count_errors += 1

    print('Errors in loading word embedddings: {}'.format(count_errors))
    return emb

