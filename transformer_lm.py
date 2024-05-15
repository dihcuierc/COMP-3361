# models.py

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random
from torch.utils.data import DataLoader


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, num_classes, num_layers, batched=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, num_positions=num_positions, batched=batched)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encode = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, mask_check=True)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        input_embedded = self.embedding(indices)
        input_embedded = self.positional_encoding(input_embedded)
        mask = torch.triu(torch.ones(indices.shape[-1], indices.shape[-1]) * float('-inf'), diagonal=1)
        output = self.encode(input_embedded, mask)
        logits = self.output_layer(output)
        return torch.log_softmax(logits[:, -1, :], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=True):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """

        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)
    

class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index

    def get_next_char_log_probs(self, context):
        self.model.eval()
        if len(context) < 20:
            context = str(" " * (20 - len(context)) + context)
        context = context[-20:]
        with torch.no_grad():
            input_indices =  torch.LongTensor([self.vocab_index.index_of(ci) for ci in context]).unsqueeze(0)
            log_prob = self.model(input_indices)
        return log_prob.squeeze(0).cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        log_prob = 0.0
        for char in next_chars:
            log_probs = self.get_next_char_log_probs(context)
            log_prob += log_probs[self.vocab_index.index_of(char)]
            context += char
        return log_prob.item()


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    def split_text_into_chunks(text, chunk_size):
        chunks = []
        for i in range(0, len(text)-20, 10):
            chunk = ' ' + text[i:i+chunk_size]
            indices =  torch.LongTensor([vocab_index.index_of(ci) for ci in chunk])
            input_indices, output_indices = indices[:20], indices[20]
            chunks.append([input_indices, output_indices])
        return chunks

    train_text_chunks = split_text_into_chunks(train_text, 20)
    dev_text_chunks = split_text_into_chunks(dev_text, 20)
    
    num_epochs = 20
    best_loss = float('inf')
    model = TransformerLanguageModel(vocab_size=27, num_positions=20, d_model=64, num_classes=27, num_layers=4)
    best_model = model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loader = DataLoader(train_text_chunks,batch_size=32,shuffle=True)
    dev_loader = DataLoader(dev_text_chunks,batch_size=32,shuffle=False)

    for epoch in range(num_epochs):
        total_train_loss = 0
        random.seed(epoch)
        model.train()
        for data in train_loader:
            logits = model(data[0])
            loss = criterion(logits, data[1])
            optimizer.zero_grad()
            loss.backward()
            total_train_loss += loss.item() * len(data[1])
            optimizer.step()
        
        with torch.no_grad():
            total_dev_loss = 0
            model.eval()
            for data in dev_loader:
                logits = model(data[0])
                loss = criterion(logits, data[1])
                total_dev_loss += loss.item()* len(data[1])
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_train_loss / (len(train_text_chunks))}, Dev Loss: {total_dev_loss / (len(dev_text_chunks))}")

        if total_dev_loss < best_loss:
            best_loss = total_dev_loss
            best_model = model
    return NeuralLanguageModel(best_model, vocab_index)
