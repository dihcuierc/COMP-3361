# transformer.py

import torch
import time
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, batched=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions, batched=batched)
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, num_classes)
        
    def forward(self, indices, mask=None):
        attention_maps = []
        input_embedded = self.embedding(indices)
        input_embedded = self.positional_encoding(input_embedded)
        output = input_embedded
        for layer in self.transformer_layers:
            output = layer(output, mask)
            attention_maps.append(output)
        logits = self.output_layer(output)
        return torch.log_softmax(logits, dim=-1), attention_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal=64, num_heads=2):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.linear = FeedForwardNetwork(d_model=d_model, d_ff=64)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_vecs, mask=None):
        # print(input_vecs.shape) 1x20x64
        attention_output = self.self_attention(input_vecs, mask)
        attention_output = self.layer_norm(input_vecs + attention_output)
        output_ff = self.linear(attention_output) + attention_output
        return self.layer_norm(output_ff)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_internal = d_model // num_heads
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
    def split_heads(self, vecs):
        if len(vecs.shape) == 2:
            seq_len, d_model = vecs.shape
            return vecs.view(seq_len, self.num_heads, self.d_internal).transpose(0, 1)
        batch, seq_len, d_model = vecs.shape
        return vecs.view(batch, seq_len, self.num_heads, self.d_internal).transpose(1, 2)
      
    def combine_heads(self, vecs):
        if len(vecs.shape)==3:
            num_heads, seq_len, self.d_internal = vecs.shape
            return vecs.transpose(0, 1).contiguous().view(seq_len, self.d_model)
        batch, num_heads, seq_len, self.d_internal = vecs.shape
        return vecs.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
          
    def forward(self, input_vecs, mask=None):
        q = self.split_heads(self.linear_q(input_vecs)) # 1x20x(64/nheads)
        k = self.split_heads(self.linear_k(input_vecs)) # 1x20x(64/nheads)
        v = self.split_heads(self.linear_v(input_vecs)) # 1x20x(64/nheads)
        
        attention_scores = (q@k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_internal)) # 1x20x20
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_prob = torch.softmax(attention_scores, dim=-1) # 1x20x20
        attention_output = attention_prob@v # 1x20x(64/nheads)
        return self.combine_heads(attention_output) # 1x20x64
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, input):
        output_ff = self.relu(self.linear1(input))
        output_ff = self.linear2(output_ff)
        
        return output_ff
        
# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
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
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


def train_classifier(args, train, dev):
    num_epochs = 10
    best_loss = float('inf')
    model = Transformer(vocab_size=27, num_positions=20, d_model=64, d_internal=64, num_classes=3, num_layers=1)
    criterion = nn.NLLLoss()
    best_model = model
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        total_train_loss = 0
        random.seed(epoch)
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)
        model.train()
        for ex_idx in ex_idxs:
            input_indices = train[ex_idx].input_tensor.unsqueeze(0)
            log_prob, _ = model(input_indices)
            output_indices = train[ex_idx].output_tensor.unsqueeze(0)
            loss = criterion(log_prob.squeeze(0), output_indices.squeeze(0))
            optimizer.zero_grad()
            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()
        ex_idxs = list(range(len(dev)))
        model.eval()
        total_dev_loss = 0
        for ex_idx in ex_idxs:
            input_indices = dev[ex_idx].input_tensor.unsqueeze(0)
            log_prob, _ = model(input_indices)
            output_indices = dev[ex_idx].output_tensor.unsqueeze(0)
            loss = criterion(log_prob.squeeze(0), output_indices.squeeze(0))
            total_dev_loss += loss.item() 
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_train_loss / len(train)}, Dev Loss: {total_dev_loss / len(dev)}")
        if total_dev_loss < best_loss:
            best_loss = total_dev_loss
            best_model = model 
    best_model.eval()
    return best_model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = True
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
                plt.close()
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))