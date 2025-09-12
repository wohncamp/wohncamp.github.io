---
layout: post
title: "Building GPT from Scratch: 1. Data Preparation"
date: 2025-09-09 22:00:00 -0700
categories: Learning
---
As a software engineer, I find that one of the most effective ways to learn new concepts is through hands-on practice. In a series of posts, I’ll document my journey of understanding how GPT works by implementing it from scratch.

Implementing and training a GPT model involves many intricate components. The intermediate results are often just numbers, and it's difficult to know if everything is on track until all the pieces are working together. In other words, when the trained model doesn’t work as expected, it's hard for a beginner to know where the problem lies. To address this, my approach is to start with a very simple, working example using expert-written libraries. I'll verify that everything is functioning as expected first, and then replace each component with my own implementation. If something isn't working, I'll know that the last updated part has an issue.

The reference model is [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M). This model is very small and thus easy to train, yet it can still generate a few coherent sentences.

In this first post, I will implement classes to prepare data for training.

Let’s start with the Tokenizer, which encodes text into token IDs and decodes token IDs back into text. The initial implementation of the tokenizer will use a pre-trained tokenizer and will be replaced by my own implementation later.

```python
# tokenizer.py

from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    def encode(self, input_string):
        return self.tokenizer.encode(input_string, return_tensors=None)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
```

I can quickly verify the implementation.

```shell
$ python
>>> from tinyllm.tokenizer import Tokenizer
>>>
>>> text = 'Hello, world!'
>>>
>>> t = Tokenizer()
>>> tokens = t.encode(text)
>>> tokens
[15496, 11, 995, 0]
>>>
>>> t.decode(tokens)
'Hello, world!'
```

Next, I will implement the dataset that supplies data to the training process. As mentioned earlier, I’ll be using the TinyStories dataset, which is available through the `datasets` package. This class will be updated later to support direct use of raw text files.

```python
# dataset.py

import torch

from datasets import load_dataset


class TinyDataset:
    def __init__(self, tokenizer, length, set_type='validation'):
        self.tokenizer = tokenizer
        self.length = length

        ds = load_dataset("roneneldan/TinyStories")
        self.data = [self.convert(d['text']) for d in ds[set_type]]

    def convert(self, text):
        """
        The returned list always has a fixed number of tokens, defined by length.
        If the tokenized result contains fewer tokens, END_OF_STRING tokens
        will be appended. Otherwise, the list will be truncated to contain
        exactly length tokens.
        """
        eos_token_id = self.tokenizer.eos_token_id

        tokens = self.tokenizer.encode(text)
        tokens += [eos_token_id] * (self.length - len(tokens))

        return tokens[:self.length]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])
```

Let's see a few data points.

```shell
$ python
>>> from tinyllm.tokenizer import Tokenizer
>>> from tinyllm.dataset import TinyDataset
>>>
>>> t = Tokenizer()
>>> ds = TinyDataset(t, 65)
>>>
>>> len(ds)
21990
>>> print(ds[0])
tensor([32565,    13, 15899,  2497,   262, 22441,  1097,   290,   531,    11,
          366, 22017,    11, 21168,    11,   534,  1097,   318,   523,  6016,
          290,  3424,  2474, 21168, 13541,   290,  8712,    11,   366, 10449,
          345,    11, 15899,    13,   314, 25245,   340,   790,  1110,   526,
          198,   198,  3260,  2712,   351,   262,  1097,    11, 21168,   290,
        15899,  2936, 47124,    13,  1119,  1043,   257,  1402, 16723,   351,
         1598,  1660,    13,  1119, 24070])
```

Now, I'm ready to implement a tiny GPT model.
