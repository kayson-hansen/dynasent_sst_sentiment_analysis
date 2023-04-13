# Description: Our model is a transformer that's a fine-tuned version of the pre-trained BERT model
# BERT-small (we tried larger BERT models, but they were too computationally expensive for our machines).
# The hyperparameters used for fine-tuning include a learning rate of 5e-5, a maximum sequence length of 512,
# a batch size of 8, and a maximum of 5 epochs with early stopping if the validation loss does not improve f
# or 5 consecutive epochs. We included evaluation of the fine-tuned models using the classification_report function
# from the sklearn library, which outputs precision, recall, and f1-scores for each class in the dataset.
# Our model achieved a macro avg f1-score of 0.577 on the Stanford Sentiment Treebank dataset, a macro avg f1-score
# of 0.716 on the DynaSent R1 dataset, and a macro avg f1-score of 0.589 on the DynaSent R2 dataset. These f1-scores
# all outperformed the baseline model given by taking the output hidden states above the [CLS] token for each
# sentence. We found representations for sentences by summing the output hidden states above each token. Our tokenization
# scheme was the one included with BERT-small. Finally, we used the TorchDeepnNeuralClassifier class as the base class
# for our model.

import torch
import torch.nn as nn
from torch_deep_neural_classifier import TorchDeepNeuralClassifier

# For the pretrained model we chose
from transformers import AutoModel
from transformers import AutoTokenizer

# Step 1: Choose pretrained model to use from hugging face
# https://huggingface.co/prajjwal1/bert-small
weights_name = "prajjwal1/bert-small"


# Step 2: Use tokenizer that comes with model
tokenizer = AutoTokenizer.from_pretrained(weights_name)


# Step 3: Get representations of tokens
def get_batch_token_ids(batch, tokenizer):
    """Map `batch` to a tensor of ids. 

    Parameters
    ----------
    batch: list of str
    tokenizer: Hugging Face tokenizer

    Returns
    -------
    dict with at least "input_ids" and "attention_mask" as keys,
    each with Tensor values

    """
    max_length = 512
    return tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt', return_attention_mask=True)


# Step 4: Define the graph for the neural network and generate representations for the sentences
class BertClassifierModule(nn.Module):
    def __init__(self,
                 n_classes,
                 hidden_activation):
        """This module loads a Transformer, adds a dense layer with activation 
        function give by `hidden_activation`, and puts a classifier
        layer on top of that as the final output. The output of
        the dense layer should have the same dimensionality as the
        model input.

        Parameters
        ----------
        n_classes : int
            Number of classes for the output layer
        hidden_activation : torch activation function
            e.g., nn.Tanh()
        weights_name : str
            Name of pretrained model to load from Hugging Face

        """
        super().__init__()
        self.n_classes = n_classes
        self.bert = AutoModel.from_pretrained(weights_name)  # v1 and v2

        self.bert.train()
        self.hidden_activation = hidden_activation
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.classifier_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.n_classes, bias=True)
        )

    def forward(self, indices, mask):
        """Process `indices` with `mask` by feeding these arguments
        to `self.bert` and then feeding the initial hidden state
        in `last_hidden_state` to `self.classifier_layer`.

        Parameters
        ----------
        indices : tensor.LongTensor of shape (n_batch, k)
            Indices into the `self.bert` embedding layer. `n_batch` is
            the number of examples and `k` is the sequence length for
            this batch
        mask : tensor.LongTensor of shape (n_batch, d)
            Binary vector indicating which values should be masked.
            `n_batch` is the number of examples and `k` is the
            sequence length for this batch

        Returns
        -------
        tensor.FloatTensor
            Predicted values, shape `(n_batch, self.n_classes)`

        """
        maskreps = self.bert(indices, attention_mask=mask)
        return self.classifier_layer(torch.sum(maskreps.last_hidden_state, dim=1))


# Step 5: Use torch_deep_neural_classifier and fine tune it on the datasets
class OriginalClassifier(TorchDeepNeuralClassifier):
    def __init__(self, *args, **kwargs):
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    def build_graph(self):
        return BertClassifierModule(
            self.n_classes_, self.hidden_activation)

    def build_dataset(self, X, y=None):
        data = get_batch_token_ids(X, self.tokenizer)
        if y is None:
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'])
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(
                data['input_ids'], data['attention_mask'], y)
        return dataset
