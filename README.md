# PyTorch re-implementation of MultiNLI baseline methods

Most scripts are from [nyu-mll/multiNLI](https://github.com/nyu-mll/multiNLI/)


# Setup

Download SNLI and MultiNLI datasets from [here](https://www.nyu.edu/projects/bowman/multinli/) and put them in `data`

```
data/
  snli_1.0/
    (files from SNLI v1.0)
  multinli_0.9/
    (files from MultiNLI v0.9)
```

Download [pretrained GloVe vectors](https://nlp.stanford.edu/projects/glove/) and put them in `data/glove.840B.300d.txt` (840B token, 300 dimensions)


# Train

```shell
cd src

# Use scene graphs
PYTHONPATH=$PYTHONPATH:. python train_snli.py cbow petModel-cbowsg --emb-train --path-train ../data/snli_1.0/snli_1.0_train.spatial.sg.jsonl --path-dev ../data/snli_1.0/snli_1.0_dev.spatial.sg.jsonl --path-test ../data/snli_1.0/snli_1.0_test.spatial.sg.jsonl --sg --gpu

# Do not use scene graphs
PYTHONPATH=$PYTHONPATH:. python train_snli.py cbow petModel-cbow --emb-train --path-train ../data/snli_1.0/snli_1.0_train.spatial.sg.jsonl --path-dev ../data/snli_1.0/snli_1.0_dev.spatial.sg.jsonl --path-test ../data/snli_1.0/snli_1.0_test.spatial.sg.jsonl --gpu
```

Note: you may want to set `OMP_NUM_THREADS=1` so that pytorch does not occupy all the available CPU cores.


# Prediction

```shell
cd src

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PYTHONPATH:. python predictions.py cbow multinli-cbow --input ../data/multinli_1.0/multinli_1.0_dev_matched.spatial.sg.jsonl
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PYTHONPATH:. python predictions.py cbow multinli-cbowsg --input ../data/multinli_1.0/multinli_1.0_dev_matched.spatial.sg.jsonl --sg
```
