---
title: "Encoder-Decoder Architecture"
seoTitle: "Encoder-Decoder Architecture"
seoDescription: "Synopsis of Encoder-decoder architecture, which is used in sequence-to-sequence tasks. Learn internal structure and training & serving these models."
datePublished: Mon Dec 11 2023 15:00:10 GMT+0000 (Coordinated Universal Time)
cuid: clq11ithp000a08jv8rbp4jei
slug: ed-arch
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1702257640626/fff52582-e7e8-4abd-8a4e-aa6d63361af0.png
tags: artificial-intelligence, machine-learning, google-cloud, google, generative-ai

---

This is a collection of notes from the [Encoder-Decoder Architecture](https://www.cloudskillsboost.google/paths/183/course_templates/543) course on Google Cloud taught by **Benoit Dherin**. Some images are taken from the course itself.

It is a detailed compilation and annotated excerpts will be available on my [**LinkedIn profile**](https://www.linkedin.com/in/akshit-keoliya/).

# Course Overview

1. Architecture
    
2. Training Phase
    
3. Serving Phase
    
4. What's Next
    

# Architecture

Encoder-Decoder is a sequence-to-sequence architecture i.e. it consumes sequences and spits out sequences.  
It performs tasks like Machine Translation, Text Summarization and Question Answering. Another example can be prompts that are given to LLM which result in response from the LLM.

![Encoder Decoder stages image from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1702255508975/471dd480-210a-4d2e-80eb-1a92aef63613.png align="center")

There are two stages: Encoder creates vector representation of input sequence, then Decoder creates sequence output.  
Both the Encoder and Decoder can have different internal architecture. These can be Recurrent Neural Networks (RNNs) or Transformer Blocks (in case of LLMs)

## Steps

![Working of Encoder-Decoder Architecture image from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1702256597838/c9427307-230d-41f1-9e0d-888f87321ccc.png align="center")

1. Each input sequence is divided into tokens.
    
2. The Encoder takes in one token at a time and produces state representing this token along with all previous tokens.
    
3. State is used in next encoding step as input for next token.
    
4. After all tokens are encoded, our output is vector representation of input sequence.
    
5. The vector representation is then passed to the Decoder.
    
6. The Decoder outputs one token at a time using current state and what it has decoded so far.
    

# Training Phase

Dataset is in the form of input-output pairs. Example: Input is sentence in source language and output is sentence in target language.

![Decoder training image from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1702255595882/0b04c6bb-8267-4633-abff-1567d80801b7.png align="center")

Model corrects weights in training based on error which is the difference between generated output for the input and the true output.

**Note:**

1. The Decoder also needs its own input at training time. Hence, we need to provide correct previous token for generating next token.  
    This process is called **Teacher Forcing** because it forces Decoder to generate token based on correct previous token and not the token it generated itself.  
    Teacher Forcing thus needs two input sentences, the original sentence (for Encoder) and left-shifted sentence (for Decoder).
    
2. Decoder only generates probability of each token in vocabulary. To select appropriate token, we can adopt following strategies:
    
    1. **Greedy Search**  
        This is the simplest strategy. We just choose the token which has the highest probability.
        
    2. **Beam Search** (BETTER)
        
        We use the probability of generated token to calculate the probability of sentence chunks. Then we keep the chunk with the highest probability.
        

# Serving Phase

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1702256649188/500aa8f2-8d43-46f1-be80-8a4c4bc47f43.png align="center")

Referring the image above, we perform following steps:

1. We feed the Encoder representation of the prompt to the Decoder along with special (Start) token that generates the first token.
    
2. Start token (GO) is represented using Embedding Layer.
    
3. Next, Recurrent layer updates the previous state produced by Encoder into new state.
    
4. This new state is passed to Dense SoftMax layer which produces word probabilities.
    
5. Finally, we select appropriate word using Greedy Search or chunk using Beam Search.
    

# What's Next?

1. The internal architectures of Encoder and Decoder model changes performance.
    
2. For Google's Attention Mechanism, RNN is replaced by Transformer Blocks.
    
    ![Transformer history image from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1702256819867/d4055ca4-0da6-4626-8bf9-7aceb17b9b70.png align="center")
    

# Ending Note

We learnt about Encoder-Decoder Architecture which is prevalent ML architecture for many sequence-to-sequence tasks.

Next, we will learn about **Attention Mechanism** that allows neural networks to focus on specific parts of input sequence and improve performance on variety of ML tasks.

Stay tuned for lots of Generative AI content!