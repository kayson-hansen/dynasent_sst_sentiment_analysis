# Description: Our model is a transformer that's a fine-tuned version of the pre-trained BERT model
# BERT-small (we tried larger BERT models, but they were too computationally expensive for our machines).
# The hyperparameters used for fine-tuning include a learning rate of 5e-5, a maximum sequence length of 512,
# a batch size of 8, and a maximum of 5 epochs with early stopping if the validation loss does not improve f
# or 5 consecutive epochs. The code also includes evaluation of the fine-tuned models using the
# classification_report function from the sklearn library, which outputs precision, recall, and f1-scores for
# each class in the dataset. Our model achieved a macro avg f1-score of 0.282 on the Stanford Sentiment Treebank
# dataset, a macro avg f1-score of 0.284 on the DynaSent R1 dataset, and a macro avg f1-score of 0.272 on the
# DynaSent R2 dataset. We found representations for sentences by averaging the embedding for the hidden layer
# above each token. Our tokenization scheme was the one included with BERT-small. Finally, we used the
# TorchDeepnNeuralClassifier class as the base class for our model.

# This file is for fitting the pytorch model to the data.
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


# Step 2: Create model
original_model = OriginalClassifier(
    hidden_activation=nn.ReLU(),
    eta=0.00005,          # Low learning rate for effective fine-tuning.
    batch_size=8,         # Small batches to avoid memory overload.
    gradient_accumulation_steps=4,  # Increase the effective batch size to 32.
    early_stopping=True,  # Early-stopping
    n_iter_no_change=5)   # params.


# Step 3: Train model on sst and dynasent datasets
_ = original_model.fit(X, y)


# Step 4: Save trained model to a file
filename = 'sum_representation_model.pth'
torch.save(original_model.build_graph().state_dict(), filename)

# Step 5: Assess model performance on test sets
sst_preds = original_model.predict(sst['validation']['sentence'])
print(classification_report(sst['validation']
      ['gold_label'], sst_preds, digits=3))
dynasent_r1_preds = original_model.predict(
    dynasent_r1['validation']['sentence'])
print(classification_report(
    dynasent_r1['validation']['gold_label'], dynasent_r1_preds, digits=3))
dynasent_r2_preds = original_model.predict(
    dynasent_r2['validation']['sentence'])
print(classification_report(
    dynasent_r2['validation']['gold_label'], dynasent_r2_preds, digits=3))
