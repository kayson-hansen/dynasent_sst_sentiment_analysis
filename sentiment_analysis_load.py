# This file is for when you are loading the pytorch model and you've already fit it to the data.

import torch
import torch.nn as nn
from model import OriginalClassifier
from datasets import load_dataset
from sklearn.metrics import classification_report


# Step 1: Load datasets
dynasent_r1 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r1.all')
dynasent_r2 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r2.all')
sst = load_dataset("SetFit/sst5")


def convert_sst_label(s):
    return s.split(" ")[-1]


for splitname in ('train', 'validation', 'test'):
    dist = [convert_sst_label(s) for s in sst[splitname]['label_text']]
    sst[splitname] = sst[splitname].add_column('gold_label', dist)
    sst[splitname] = sst[splitname].add_column(
        'sentence', sst[splitname]['text'])


X = dynasent_r1['train']['sentence'] + \
    dynasent_r2['train']['sentence'] + sst['train']['sentence']
y = dynasent_r1['train']['gold_label'] + \
    dynasent_r2['train']['gold_label'] + sst['train']['gold_label']


# Step 2: Load model
saved_model = OriginalClassifier(
    hidden_activation=nn.ReLU(),
    eta=0.00005,          # Low learning rate for effective fine-tuning.
    batch_size=8,         # Small batches to avoid memory overload.
    gradient_accumulation_steps=4,  # Increase the effective batch size to 32.
    early_stopping=True,  # Early-stopping
    n_iter_no_change=5)   # params.

filename = 'sum_representation_model.pkl'
saved_model.from_pickle(filename)

saved_model.build_dataset(X, y)
saved_model.initialize()


# Step 3: Make predictions and assess model performance
sst_preds = saved_model.predict(sst['validation']['sentence'])
print(classification_report(sst['validation']
      ['gold_label'], sst_preds, digits=3))
dynasent_r1_preds = saved_model.predict(dynasent_r1['validation']['sentence'])
print(classification_report(
    dynasent_r1['validation']['gold_label'], dynasent_r1_preds, digits=3))
dynasent_r2_preds = saved_model.predict(dynasent_r2['validation']['sentence'])
print(classification_report(
    dynasent_r2['validation']['gold_label'], dynasent_r2_preds, digits=3))
