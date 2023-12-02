---
title: "Introduction to Generative AI (Part 1)"
seoTitle: "Introduction to Generative AI"
seoDescription: "Basic introduction to Generative Artificial Intelligence from Google Cloud course."
datePublished: Thu Nov 30 2023 15:30:10 GMT+0000 (Coordinated Universal Time)
cuid: clplcr0mn000008kzebrs8g13
slug: introduction-to-generative-ai-part-1
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1701271991900/daf61719-9a24-4de6-aa3b-81d088d5b39f.png
tags: artificial-intelligence, cloud, machine-learning, google-cloud, google, transformers, generative-ai

---

This is collection of notes from the [Generative AI course](https://www.cloudskillsboost.google/course_templates/536) on Google Cloud taught by Dr. Gwendolyn Striping. Some images are taken from the course itself.

It is a detailed compilation and annotated excerpts will be available on my [LinkedIn profile](https://www.linkedin.com/in/akshit-keoliya/).

# Course Overview

The course is divided into four parts. (We will cover first two in this section)

1. Defining Generative AI
    
2. Working of Generative AI
    
3. Generative AI Model Types
    
4. Generative Applications.
    

# Defining Generative AI

Course gives us an overview of what Artificial Intelligence is before defining Generative AI.  
AI is a Computer Science branch (discipline like physics) that builds intelligent systems that can reason, learn and act autonomously. They perform the actions that usually require human intelligence.

## AI vs ML vs DL

![AI vs ML image taken from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701267623268/5281cedd-778f-4ba2-9262-64d64e5c6ba7.png align="center")

Machine learning is a subfield of AI that gives system the ability to learn without explicit programming.  
Now, ML is divided into 3 types:

1. **Supervised Learning**  
    We have examples and their correct labels. A label can be a name, tag or a number. It detects pattern in data to predict future values.
    
2. **Unsupervised Learning**  
    We only have examples and no labels. It is used for discovering structure in data to form groups.
    
3. **Semi-supervised Learning**  
    From the labelled data, we learn the basic concept of the task. Using the remaining unlabeled data, we generalize to new examples.
    

Deep Learning is subfield of ML that uses Artificial Neural Networks consisting of multiple layers of Neurons to process complex patterns.  
Now, DL models can be classified into two types:

1. **Discriminative Models**  
    These are used for classification or prediction tasks. They typically work on labelled data and learn label-example relationship.
    
2. **Generative Models**  
    These generate new data from the learnt probability distribution of existing data. They perform tasks like predicting next word or pixel in the sequence.
    

## Generative AI

![Traditional Programming and Neural Networks image taken from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701263102893/5c12885d-644a-44e3-b42b-b09aacc0c43a.png align="center")

We have come a long way from Traditional Programming that required handcrafted rules, then Neural Networks that can learn how to distinguish examples and now to Generative AI that can create its own content.

![Generative AI Image taken from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701263138282/5ef01974-6869-4a70-b810-3ff0533e32e9.png align="center")

As mentioned above, Generative AI is a subset of DL that can generate new content based on existing content.  
The learning process is called Training. It creates a statistical model, which when given a suitable prompt, predicts expected outcome which is new content.  
Generative AI output can be Natural Language, Image, Audio, Code, etc. Its output can never be a discrete number, class or probability.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701264179410/e90329ee-4c3c-4eca-b79e-15d298bd6351.png align="center")

**Generative Language Models** learn pattern in language and given some text, predicts what comes next.  
**Generative Image Models** produce new image from random noise/prompt using techniques like diffusion.

# Working of Generative AI

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701264352612/392ec9f6-45ff-4297-8776-e54ac1d0767d.png align="center")

The Transformer model is the core of Generative AI workflow. The entire process is as follows:

1. Transformer takes in input which is received by its Encoder component. This component encodes the input into fixed sized representation.
    
2. The representation is passed to Decoder Component, which decodes it based on the specific task.
    
3. This decoded representation is passed to Generative Pre-trained Transformer Model which is trained in unsupervised fashion on large data and has billions of parameters.
    
4. Pre-trained Transformer then generates output based on the decoded representation.
    

## Hallucinations

Hallucinations in Generative AI models are defined as outputs that are either non-sensical, incorrect or misleading.  
These can be caused by not having enough data, context or constraints on the model. Training on noisy or dirty data can also cause model to hallucinate.

# Ending Note

We will be covering the Model Types in Generative AI and their applications in the next post. More details about Transformer models will also be covered in subsequent posts. Stay tuned for lots of Generative AI content!