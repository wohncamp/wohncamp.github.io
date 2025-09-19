---
layout: post
title: "Building GPT from Scratch: 5. LayerNorm"
date: 2025-09-18 22:00:00 -0700
categories: Learning
math: true
---

The next component I want to replace is `GPTNeoBlock`. Since it’s not a single layer but a composition of several layers, I plan to copy its implementation and replace each layer individually. I can simply copy the GPTNeoBlock implementation from [Github](https://raw.githubusercontent.com/huggingface/transformers/e3cc4487fe66e03ec85970ea2db8e5fb34c455f4/src/transformers/models/gpt_neo/modeling_gpt_neo.py).

```python
# model.py

from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoAttention, GPTNeoMLP


class GPTBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, epsilon=config.layer_norm_epsilon)
        self.attn = GPTNeoAttention(config, layer_id)
        self.ln_2 = nn.LayerNorm(hidden_size, epsilon=config.layer_norm_epsilon)
        self.mlp = GPTNeoMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)



```

After that, I can substitute `GPTNeoBlock` with `GPTBlock`.

```diff
-        self.blocks = nn.ModuleList([GPTNeoBlock(config, layer_id=i) for i in range(config.num_layers)])
+        self.blocks = nn.ModuleList([GPTBlock(config, layer_id=i) for i in range(config.num_layers)])
```

The first component I will replace is nn.LayerNorm. Layer normalization was introduced in the [Layer Normalization](https://arxiv.org/abs/1607.06450) paper. In short, it is known to make training more stable. The concept is straightforward: LayerNorm normalizes each input (i.e., the output of the previous layer) using the following normalization terms.

$$
\mu = \frac{1}{H} \sum_{i=1}^{H}a_i \hspace{1cm} \sigma = \sqrt{\frac{1}{H} \sum_{i=1}^{H} (a_i - \mu)^2}
$$

Then, the input can be normalized as follows.

$$
x_{normalized} = \frac{x - \mu}{\sigma}
$$

In practice, to avoid division-by-zero errors, it’s common to add a small constant to the denominator. Therefore,

$$
x_{normalized} = \frac{x - \mu}{\sqrt{\sigma ^ 2 + \epsilon}}
$$

Now let's implement our own `LayerNorm`.

```python
# norm.py

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()

        self.epsilon = epsilon

        # This is required to send this module to GPU.
        # PyTorch complains to send a module without any Parameters or Buffers to GPU.
        self.register_buffer("dummy", torch.empty(0))

    def forward(self, X):
        mu = torch.sum(X, dim=-1, keepdim=True) / X.shape[-1]
        variance = torch.sum(torch.square(X - mu), dim=-1, keepdim=True) / X.shape[-1]

        return (X - mu) / torch.sqrt(variance + self.epsilon)
```

I can test the implementation by comparing its output with that of PyTorch’s `LayerNorm`.

```shell
>>> import torch
>>> from tinyllm import norm
>>> 
>>> X = torch.randn(2, 3, 4)
>>> 
>>> norm.LayerNorm()(X)
tensor([[[ 0.2828,  1.4225, -1.3231, -0.3822],
         [-0.5129, -1.2819,  0.4060,  1.3888],
         [-0.1348, -0.1960,  1.5598, -1.2290]],

        [[ 1.6377, -0.3495, -1.0720, -0.2162],
         [ 1.1558, -1.1942, -0.7673,  0.8058],
         [-1.5553,  0.3317,  0.0108,  1.2128]]])
>>> torch.nn.LayerNorm(4)(X)
tensor([[[ 0.2828,  1.4225, -1.3231, -0.3822],
         [-0.5129, -1.2819,  0.4060,  1.3888],
         [-0.1348, -0.1960,  1.5598, -1.2290]],

        [[ 1.6377, -0.3495, -1.0720, -0.2162],
         [ 1.1558, -1.1942, -0.7673,  0.8058],
         [-1.5553,  0.3317,  0.0108,  1.2128]]],
       grad_fn=<NativeLayerNormBackward0>)
```

As I’ve verified, the custom LayerNorm implementation produces the same results as PyTorch’s `nn.LayerNorm`. Now, I can replace `nn.LayerNorm` with `norm.LayerNorm` in TinyGPT.

```diff
-        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
+        self.ln_1 = norm.LayerNorm()
         self.attn = GPTNeoAttention(config, layer_id)
-        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
+        self.ln_2 = norm.LayerNorm()
```

That wraps up the replacement of LayerNorm. In PyTorch’s implementation, there are two parameters, $$ \gamma $$ and $$ \beta $$, which serve as scale and shift factors, respectively. These are omitted in `norm.LayerNorm`. I haven’t read it yet, but [Understanding and Improving Layer Normalization](https://proceedings.neurips.cc/paper_files/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf) might offer deeper insights into layer normalization and the role of these parameters.

So far, I’ve replaced a few simple modules, and it’s already been a great learning experience with PyTorch. In the next post, I’ll implement Attention, which is the core of the Transformer, and hence the GPT architecture.
