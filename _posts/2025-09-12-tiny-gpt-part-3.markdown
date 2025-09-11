---
layout: post
title: "Building GPT from Scratch: 3. Training"
date: 2025-09-10 22:00:00 -0700
categories: Learning
---

I will use GPT-Neo to implement a lightweight version of GPT. GPT-Neo requires a configuration instance that includes hyperparameters such as context size and number of layers. Although GPT-Neo provides its own configuration class, I’m defining a custom `Config` class to make future replacements easier. I’ve copied most of the non-trivial values from GPT-Neo, while assigning smaller values to other hyperparameters to keep the model compact.

```python
# model.py

class Config:
    def __init__(self):
        self.hidden_size = 64
        self.embedding_size = 64
        self.context_size = 64
        self.num_layers = 8
        self.layer_norm_epsilon = 1e-05
        self.attention_layers = ['local'] * self.num_layers
        self.max_position_embeddings = 2048
        self.window_size = self.hidden_size
        self.attention_dropout = 0.0
        self.resid_dropout = 0.0
        self.num_heads = 16
        self._attn_implementation = 'eager'
        self.intermediate_size = None
        self.activation_function = 'gelu_new'
```

Now it's finally time to implement the `GPT` class. The implementation is quite simple. The model architecture is defined in the constructor and is fairly straightforward. The `forward` method is also simple — it just passes the input tensor through each component and returns the result. The only potentially non-obvious part is the positional embedding, but I’ll dive into that in more detail later.

```python
# model.py (continued)

import torch
import torch.nn as nn

from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock


class GPT(nn.Module):
    def __init__(self, tokenizer, config):
        super().__init__()

        self.tokenizer = tokenizer
        self.config = config

        self.word_token_embedding = nn.Embedding(self.tokenizer.vocab_size, config.embedding_size)
        self.word_position_embedding = nn.Embedding(config.context_size, config.embedding_size)
        self.blocks = nn.ModuleList([GPTNeoBlock(config, layer_id=i) for i in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.embedding_size, self.tokenizer.vocab_size, bias=False)

    def forward(self, X):
        X = X[:, :self.config.context_size]

        position_ids = torch.arange(X.shape[1]).unsqueeze(0).expand(X.shape[0], -1).to(X.device)
        X = self.word_token_embedding(X) + self.word_position_embedding(position_ids)

        for block in self.blocks:
            X = block(X)[0]

        X = self.norm(X)
        X = self.output(X)

        return X

    def next_token(self, prompt):
        tokens = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        logits = self.forward(tokens.to(self.device))
        next_token = torch.argmax(logits, dim=-1).squeeze(0)[-1].item()

        return self.tokenizer.decode([next_token])
    
    @property
    def device(self):
        return next(self.parameters()).device
```

Since GPT’s goal is to generate the next token given an input, I added a `next_token` method to the `GPT` class. This method takes a `prompt` as input, converts it into tokens, feeds them into the model, and selects the next token based on the output.

Now, let's see if it works well.

```shell
$ python
>>> from tokenizer import Tokenizer
>>> from model import Config, GPT
>>>
>>> t = Tokenizer()
>>> gpt = GPT(t, Config())
>>>
>>> gpt.next_token('Once upon a')
' Delhi'
>>> gpt.next_token('Once upon a Delhi')
' Marathon'
>>> gpt.next_token('Once upon a Delhi Marathon')
' Compact'
```

It generates a sentence like `Once upon a Delhi Marathon Compact`, which doesn't make much sense. This is expected, since the model hasn't been trained yet. The next step will be to implement a training process.
