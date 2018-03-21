from torch import nn
from torch.nn import functional as F
import torch

class CBOW(nn.Module):
    def __init__(self, hidden_dim, embeddings, keep_rate):
        super(CBOW, self).__init__()

        ## Define hyperparameters
        self.embedding_dim = embeddings.shape[1]
        self.dim = hidden_dim
        self.dropout_rate = 1.0 - keep_rate

        ## Define remaning parameters 
        self.E = nn.Embedding(embeddings.shape[0], self.embedding_dim, padding_idx=0)
        self.E.weight.data.copy_(torch.from_numpy(embeddings))
        self.W_0 = nn.Linear(self.embedding_dim * 4, self.dim)
        self.W_1 = nn.Linear(self.dim, self.dim)
        self.W_2 = nn.Linear(self.dim, self.dim)
        self.W_cl = nn.Linear(self.dim, 3)

    def forward(self, premise_x, hypothesis_x, train=True):
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

        ## Combinations
        h_diff = premise_rep - hypothesis_rep
        h_mul = premise_rep * hypothesis_rep

        ## MLP
        mlp_input = torch.cat([premise_rep, hypothesis_rep, h_diff, h_mul], 1)
        h_1 = self.W_0(mlp_input)
        h_2 = self.W_1(h_1)
        h_3 = self.W_2(h_2)
        if train:
            h_drop = F.dropout(h_3, p=self.dropout_rate)
        else:
            h_drop = h_3

        # Get prediction
        return self.W_cl(h_drop)
