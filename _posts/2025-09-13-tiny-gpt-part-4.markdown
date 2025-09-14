---
layout: post
title: "Building GPT from Scratch: 4. Embedding"
date: 2025-09-13 09:00:00 -0700
categories: Learning
---

When I look at the constructor of the GPT class, I notice that the first two modules are `Embedding`. So, what exactly is embedding?

Neural networks, including large language models (LLMs), require input in the form of numerical vectors. However, the data we use for LLMs is typically a **sequence** of **words**. Embedding is a method for converting these sequences of words into numerical vectors.

Let’s start with the *words*: how can we convert words into numerical vectors? One of the simplest approaches is called one-hot encoding. Suppose our dictionary contains only three words: apple, orange, and banana. We can represent each word as a three-dimensional vector—apple as `[1, 0, 0]`, orange as `[0, 1, 0]`, and banana as `[0, 0, 1]`. These vectors can then be fed into the LLM.

However, one-hot encoding has a major drawback: it doesn’t capture relationships between words. Mathematically, all the vectors are orthogonal. But we know that apple, orange, and banana share something in common: they're all fruits! And they likely have little in common with words like galaxy or black hole.

Embedding addresses this limitation. In one-hot encoding, each word is represented by a vector whose dimension equals the size of the dictionary. In contrast, embedding uses vectors of arbitrary dimension (a hyperparameter), and the values in these vectors are learned from data. One intuitive way to learn embeddings is by using neighboring words: for each word in a corpus, one can identify its neighboring words and train a shallow network to learn its embedding vector.

In LLMs, it's more common to let the model learn embeddings during training rather than using pre-trained ones. Our tiny GPT model follows this approach as well. Therefore, implementing embedding layers without training is sufficient.

The next part is *sequence*. Words can have different meanings depending on where they appear in a sentence. A simple approach to handle this is to provide additional information based on their position: this is called Position Embedding. Regardless of the word itself, positional embedding assigns the same values to any word that appears in the same position.

If the embedding dimension is `N`, then there is a positional embedding vector of dimension `N` for the 0th position in the sentence. This vector is added to the word's embedding. The same process continues for every position in the sentence.

Personally, positional embedding feels a bit odd to me - it seems to oversimplify the complexity of language grammar. Maybe this could be an interesting topic for further research. That said, in practice, positional embedding is known to work quite well.

Let’s go ahead and implement two embeddings.

```python
# embedding.py

import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, dim_embedding):
        super().__init__()

        self.embedding = nn.Parameter(torch.randn(vocab_size, dim_embedding))

    def forward(self, X):
        return self.embedding[X, :]


class PositionEmbedding(nn.Module):
    def __init__(self, max_length, dim_embedding):
        super().__init__()

        self.embedding = nn.Parameter(torch.randn(max_length, dim_embedding))

    def forward(self, X):
        return self.embedding[torch.arange(X.shape[1]).repeat(X.shape[0], 1), :]
```

Implementing embedding is straightforward. In the `Embedding` class, I created a random tensor with dimensions `(vocab_size, dim_embedding)`. When input data arrives, it selects the corresponding embedding vector for each token and returns them.

`PositionEmbedding` works similarly, but instead of using the actual data `X`, it relies on the shape of `X`. It generates a 0-based index sequence with the same length as the input (`X.shape[1]`), and repeats it across the batch dimension (`X.shape[0]`). Then, just like in Embedding, it slices the positional embedding parameters and returns the result.

Now, I need to update the `GPT` model to use our custom embedding layers.


```diff
-        self.word_token_embedding = nn.Embedding(self.tokenizer.vocab_size, config.embedding_size)
-        self.word_position_embedding = nn.Embedding(config.context_size, config.embedding_size)
+        self.word_token_embedding = embedding.WordEmbedding(self.tokenizer.vocab_size, config.embedding_size)
+        self.word_position_embedding = embedding.PositionEmbedding(config.context_size, config.embedding_size)



-        position_ids = torch.arange(X.shape[1]).unsqueeze(0).expand(X.shape[0], -1).to(X.device)
-        X = self.word_token_embedding(X) + self.word_position_embedding(position_ids)
+        X = self.word_token_embedding(X) + self.word_position_embedding(X)
```

I can run the training process again to verify that my embedding implementation is working as expected. For now, training with the validation set should be sufficient.


```shell
$ python 
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
Loss: 3.3705: 100%|███████████████████████████████████████████████| 688/688 [15:20<00:00,  1.34s/it]
Epoch 0: 3.3704607486724854
>>> text = 'Once upon a'
>>> for i in range(10):
...     text += gpt.next_token(text)
...
>>> print(text)
Once upon a time, there was a little girl named Lily.
```

That's it! I was able to replace the embedding layer with my own implementation.
