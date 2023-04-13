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
    """Map `batch` to a tensor of ids. The return
    value should meet the following specification:

    1. The max length should be 512.
    2. Examples longer than the max length should be truncated
    3. Examples should be padded to the max length for the batch.
    4. The special [CLS] should be added to the start and the special 
       token [SEP] should be added to the end.
    5. The attention mask should be returned
    6. The return value of each component should be a tensor.    

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
        # Add the new parameters here using `nn.Sequential`.
        # We can define this layer as
        #
        #  h = f(cW1 + b_h)
        #  y = hW2 + b_y
        #
        # where c is the final hidden state above the [CLS] token,
        # W1 has dimensionality (self.hidden_dim, self.hidden_dim),
        # W2 has dimensionality (self.hidden_dim, self.n_classes),
        # and we rely on the PyTorch loss function to add apply a
        # softmax to y.
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
