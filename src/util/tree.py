#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import LongTensor
from torch.autograd import Variable

class Tree():
    """Scene graph tree."""
    def __init__(self, word_idx):
        self.parent = None
        self.word_idx = word_idx
        self.num_children = 0
        self.children = []

    def add_child(self, child, relation_id):
        child.parent = self
        self.num_children += 1
        self.children.append((child, relation_id))

def seq2tree(seq):
    """Convert a sequence into a tree."""
    heads = []
    trees = [Tree(i) for i in range(len(seq))]
    for tree, item in zip(trees, seq):
        par_id, rel_id = item[2], item[3]
        if par_id == 0:  # root
            heads.append(tree)
            continue
        if par_id < 0:
            continue
        try:
            trees[par_id - 1].add_child(tree, relation_id=rel_id)
        except IndexError:
            pass
    root = Tree(-1)  # root
    for head in heads:
        root.add_child(tree, relation_id=-1)
    return root
