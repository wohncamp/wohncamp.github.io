---
layout: post
title: "Building GPT from Scratch: 6. Attention"
date: 2025-09-23 22:00:00 -0700
categories: Learning
math: true
---

Finally, attention - the core of Transformer architecture.

Basically speaking, attention mechanism is to identify the importance of neighbor words for the given word, and use the information in processing. For example, when translating English to Korean, simple word-by-word conversion doesn't work. The translator needs to identify related words from the source sentences, and chooses right words in the target language. Attention mechanism is one way to identify this information.

[Transformer paper](https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) explains the self-attention mechanism with query, key, and value, but it was little hard for me to intuitively understand it. I searched a few explanation over internet, and following one was how I convinced myself. It might be easier to understand, but might not be very strictly correct, so please keep it in mind.

Assume that a sentence "I play basketball" is given. To understand the word "play", the algorithm might need to pay more *attention* to "basketball", as "play" can mean different things: playing other sports, or even instruments, not sports. If word embeddings are as expected, "play" and "basketball" can have close embeddigs (note: "play" and "piano" can have similar distance too), while "I" and "play" can have relatively far embeddings. That means, if I calculate dot products between those embeddings, the dot product of "play" and "backetball" will have a bigger value than the dot product of "play" and "I".

And these dot products can be interpreted as importance, or *attention scores*, to underastnd the word "play". They tell us the importance of each word in the given sentence to understand the word "play". Since there are three words in the sentence (I play basketball), for the word "play", there can be three attention scores.

* play x I: low
* play x play: highest
* play x basketball: high

And *attention weights* are normalized attention scores.

The new representation of the word "play" with attention is the attention-weighted sum of embeddings, i.e., $$ w_I \times $$ I + $$ w_{play} \times $$ play + $$ w_{basketball} \times $$ basketball.

Many papers and articles use terminologies like query, key, and value. In my example, since I wanted to get a new representation of "play", the "play" is the query - the word in question. Keys are all words in the sentence ("I", "play", and "basketball"). By calculating dot products between query and each key, I can get scores of each key against query, higher is better. These scores will help me select values for the query, where values are again the words in the sentence (so keys and values are same). Instead of choosing only one value with the highest weight, I can generate a new value by using weighted sum. This is the new representation of the query, "play", with attention.

However, above approach is deterministic, or not trainable. Once word embedding is fixed, attention scores are fixed, and the new representation is also fixed. There is no learnable parameters. To make it trainable, we can introduce weights for query, key, and value. Instead of using the embedding of "play" directly to calculate dot products with "I", "play", and "basketball", I can multiply the embedding of "play" with query weights, which are trainable parameters, and also multiply embeddings of "I", "play", and "basketball" with key weights, also trainable parameters. It converts attention weights from deterministic values to learnable parameters. Similarly, values are also multiplied by value weights. Again, these weights allow GPT model to learn where to pay attention while understanding a give word from training data.

```python
# attention.py

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size, head_size, eps=1e-10):
        super().__init__()

        self.w_q = self.get_tensor(hidden_size, hidden_size)
        self.w_k = self.get_tensor(hidden_size, hidden_size)
        self.w_v = self.get_tensor(hidden_size, head_size)

        self.eps = eps

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        query = X @ self.w_q
        key = X @ self.w_k
        value = X @ self.w_v

        attention_scores = query @ torch.transpose(key, 1, 2)
        attention_weights = self.softmax(attention_scores / key.shape[-1] ** 0.5)

        context_length = attention_weights.shape[1]
        casual_attention_mask = torch.tril(torch.ones(context_length, context_length, device=attention_weights.device))

        casual_attention = attention_weights * casual_attention_mask
        casual_attention = casual_attention / (torch.sum(casual_attention, dim=-1, keepdim=True) + self.eps)

        return casual_attention @ value

    def get_tensor(self, num_rows, num_columns):
        tensor = torch.empty(num_rows, num_columns, requires_grad=True)
        nn.init.kaiming_normal_(tensor)

        return nn.Parameter(tensor)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        if hidden_size % num_heads:
            raise Exception(f'Invalid hidden_size and num_heads. hidden_size: {hidden_size}, num_heads: {num_heads}, remaining: {hidden_size % num_heads}')

        self.attentions = nn.ModuleList([Attention(hidden_size, hidden_size // num_heads) for i in range(num_heads)])

    def forward(self, X, **kwargs):
        output = [attention.forward(X) for attention in self.attentions]

        return torch.cat(output, dim=-1)
```

Now I should replace attention implementation.

```diff
-        self.attn = GPTNeoAttention(config, layer_id)
+        self.attn = attention.MultiHeadAttention(config.hidden_size, config.num_heads)
```

Since my `MultiHeadAttention` returns results in the slightly different form, I had to update `model.py` as well.

```diff
-        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
-        outputs = attn_outputs[1:]
+        attn_output = attn_outputs
```

```diff
-        if use_cache:
-            outputs = (hidden_states,) + outputs
-        else:
-            outputs = (hidden_states,) + outputs[1:]
-
-        return outputs  # hidden_states, present, (attentions, cross_attentions)
+        return (hidden_states, )
```

As before, I could validate the implementation by training the model with small dataset.


```shell
>>> import torch
>>> from torch.utils.data import DataLoader
>>>
>>> from tinyllm.tokenizer import Tokenizer
>>> from tinyllm.dataset import TinyDataset
>>> from tinyllm.model import GPT, Config
>>> from tinyllm.train import Trainer
>>>
>>> config = Config()
>>> batch_size=32
>>>
>>> tokenizer = Tokenizer()
>>> dataset = TinyDataset(tokenizer, config.context_size + 1)
>>> data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
>>>
>>> gpt = GPT(tokenizer, config)
>>> trainer = Trainer(gpt, data_loader, epochs=1)
>>> trainer.train()
Loss: 3.8433: 100%|███████████████████████████████████████████████| 688/688 [16:33<00:00,  1.44s/it]
Epoch 0: 3.843294382095337
>>> with torch.no_grad():
...     text = 'Once upon a'
...     for i in range(10):
...         text += gpt.next_token(text)
...
>>> print(text)
Once upon a time, there was a little girl named Lily.
```
