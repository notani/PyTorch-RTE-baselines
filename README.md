# PyTorch re-implementation of MultiNLI baseline methods

Most scripts are from [nyu-mll/multiNLI](https://github.com/nyu-mll/multiNLI/)


# Setup

Download SNLI and MultiNLI datasets from [here](https://www.nyu.edu/projects/bowman/multinli/) and put them in `data`

```
data/
  snli_1.0/
    (files from SNLI v1.0)
  multinli_0.9/
    (files from SNLI v0.9)
```

Download [pretrained GloVe vectors](https://nlp.stanford.edu/projects/glove/) and put them in `data/glove.840B.300d.txt` (840B token, 300 dimensions)


# Train

```shell
cd src
PYTHONPATH=$PYTHONPATH:. python train_snli.py cbow petModel-0 --keep_rate 0.9 --seq_length 25 --emb_train --gpu
```

or if you use GPU,

```
cd src
PYTHONPATH=$PYTHONPATH:. python train_snli.py cbow petModel-0 --keep_rate 0.9 --seq_length 25 --emb_train
```
