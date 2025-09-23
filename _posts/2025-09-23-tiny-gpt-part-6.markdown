---
layout: post
title: "Building GPT from Scratch: 6. Attention"
date: 2025-09-23 22:00:00 -0700
categories: Learning
math: true
published: true
---

Finally, attention - the core of Transformer architecture.

Basically speaking, attention mechanism is to identify the importance of neighbor words for the given word, and use the information in processing. For example, when translating English to Korean, simple word-by-word conversion doesn't work. The translator needs to identify related words from the source sentences, and chooses right words in the target language. Attention mechanism is one way to identify this information.

[Transformer paper](https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) explains the self-attention mechanism with query, key, and value, but it was little hard for me to intuitively understand it. I searched a few explanation over internet, and following one was how I convinced myself. It might be easier to understand, but might not be very strictly correct, so please keep it in mind.

Assume that a sentence "I play basketball" is given. To understand the word "play", the algorithm might need to pay more *attention* to "basketball", as "play" can mean different things: playing other sports, or even instruments, not sports. If word embeddings are as expected, "play" and "basketball" can have close embeddigs (note: "play" and "piano" can have similar distance too), while "I" and "play" can have relatively far embeddings. That means, if I calculate dot products between those embeddings, the dot product of "play" and "backetball" will have a bigger value than the dot product of "play" and "I".

And these dot products can be interpreted as importance, or *attention scores*, to underastnd the word "play". They tell us the importance of each word in the given sentence to understand the word "play". Since there are three words in the sentence (I play basketball), for the word "play", there can be three attention scores.

* $$ e_{play} \cdot e_{I} $$ : low
* $$ e_{play} \cdot e_{play} $$ : highest
* $$ e_{play} \cdot e_{basketball} $$ : high

where $$ e $$ is an embedding, e.g., $$ e_{play} $$ is an embedding of the word "play".

At last, we can caluclate *attention weights* by normalizing attention scores.

The new representation of the word "play" with attention is the attention-weighted sum of embeddings, i.e., $$ w_{play:I} \cdot e_{I} + w_{play:play} \cdot e_{play} + w_{play:basketball} \cdot e_{basketball} $$.

Many papers and articles use terminologies like query, key, and value. In my example, since I wanted to get a new representation of "play", the "play" is the query - the word in question. Keys are all words in the sentence ("I", "play", and "basketball"). By calculating dot products between query and each key, I can get scores of each key against query, higher is better. These scores will help me select values for the query, where values are again the words in the sentence (so keys and values are same). Instead of choosing only one value with the highest weight, I can generate a new value by using weighted sum. This is the new representation of the query, "play", with attention.

However, above approach is deterministic. Once word embedding is fixed, attention scores are fixed, and the new representation is also fixed. There is no learnable parameters. To make it trainable, we can introduce weights for query, key, and value. Instead of using the embedding of "play" directly to calculate dot products with "I", "play", and "basketball", I can multiply the embedding of "play" with query weights, which are trainable parameters, and also multiply embeddings of "I", "play", and "basketball" with key weights, also trainable parameters. It converts attention weights from deterministic values to learnable parameters. Similarly, values are also multiplied by value weights. Again, these weights allow GPT model to learn where to pay attention while understanding a give word from training data.
