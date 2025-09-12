---
layout: post
title: "Building GPT from Scratch: 3. Training"
date: 2025-09-12 09:00:00 -0700
categories: Learning
---

Training the tiny GPT model is relatively straightforward. For each data point, the training process splits it into `X` and `Y`, where `Y` is simply `X` shifted by one token. This means the model is trained to predict the next token at each position. As a result, the final token in the output represents the model’s prediction for the next token in the sequence.

In addition to the core training logic, I’ve integrated the tqdm package to monitor progress during training. The model also saves its parameters after each epoch.

```python
# train.py

import torch

from tqdm import tqdm


class Trainer:
    def __init__(self, model, data_loader, epochs=5, device='cpu'):
        self.model = model
        self.data_loader = data_loader
        self.epochs = epochs
        self.device = device

    def train(self):
        self.model.to(self.device)
        self.model.train()

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(self.epochs):
            pbar = tqdm(total=len(self.data_loader))
            for d in self.data_loader:
                X = d[:, :-1].to(self.device)
                Y = d[:, 1:].to(self.device)

                output = self.model(X)

                loss = loss_fn(output.transpose(1, 2), Y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.update(1)
                pbar.set_description(f'Loss: {loss:.4f}')

            pbar.close()

            print(f'Epoch {epoch}: {loss}')
            torch.save(self.model.state_dict(), f'./{epoch + 1}.torch')
```

Now let's run the training process.

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
>>>
>>> data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
>>>
>>> gpt = GPT(tokenizer, config)
>>> trainer = Trainer(gpt, data_loader, epochs=1)
>>>
>>> trainer.train()
Loss: 2.7894: 100%|███████████████████████████████████████████████| 688/688 [14:47<00:00,  1.29s/it]
Epoch 0: 2.789372444152832
>>>
>>> text = 'Once upon a'
>>> for i in range(10):
...     text += gpt.next_token(text)
...
>>> print(text)
Once upon a time, there was a little girl named Lily.
```

Most likely, the model is overfitted due to the small amount of training data (I used the validation set). However, processing 688 batches took about 15 minutes on my MacBook Air, while the full training set contains 66,242 batches. Roughly estimated, training could take around a day. So, I used a `g6.2xlarge` instance on AWS to speed up the process.

Here is the result from `g6.2xlarge` instance. The training took about an hour.

```shell
$ python3
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
>>> dataset = TinyDataset(tokenizer, config.context_size + 1, set_type='train')
>>> data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
>>>
>>> gpt = GPT(tokenizer, config)
>>> trainer = Trainer(gpt, data_loader, epochs=1, device='cuda')
>>>
>>> trainer.train()
Loss: 1.8627: 100%|███████████████████████████████████████████| 66242/66242 [53:58<00:00, 20.46it/s]
Epoch 0: 1.8627163171768188
>>>
>>> text = 'Once upon a'
>>> for i in range(36):
...   text += gpt.next_token(text)
...
>>> print(text)
Once upon a time, there was a little girl named Lily. She loved to play outside and explore the world around her. One day, she found a big, shiny rock on the ground.
```

Now I have a working GPT model with training tools. It is time to implement each component from scratch to enhance my understanding.
