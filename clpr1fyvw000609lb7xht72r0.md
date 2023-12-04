---
title: "Introduction to Large Language Models"
seoTitle: "Introduction to Large Language Models"
seoDescription: "Basic Introduction to Large Language Models, their use cases, types and prompt tuning."
datePublished: Mon Dec 04 2023 15:00:15 GMT+0000 (Coordinated Universal Time)
cuid: clpr1fyvw000609lb7xht72r0
slug: intro-to-llms
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1701529136947/ab33b431-c2dc-48eb-870c-792784ee02b2.png
tags: artificial-intelligence, machine-learning, google-cloud, google, transformers, llm, generative-ai, promptengineering

---

This is collection of notes from the [Introduction to Large Language Models course](https://www.cloudskillsboost.google/course_templates/539) on Google Cloud taught by **John Ewald**. Images are taken from the course itself.

It is a detailed compilation and annotated excerpts will be available on my [**LinkedIn profile**](https://www.linkedin.com/in/akshit-keoliya/).

# Course Overview

The course is divided into 4 parts:

1. Define Large Language Models (LLMs)
    
2. LLM Use Cases
    
3. Prompt Tuning and Tuning LLMs
    
4. Generative AI development tools.
    

# Defining Large Language Models (LLMs)

Large Language Models, as the name suggests, are large general purpose language models that can be pretrained and fine-tuned.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701528142996/4f92391e-9cc8-4fe9-b355-e620137559e3.png align="center")

Pretraining is the process to teach the LLMs to perform basic tasks such as Text Classification, Question Answering (QA), Document Summarization and Text Generation.

These LLMs can be tailored to add upon domain specific tasks. These domains can be Retail, Finance, Entertainment, etc.

## Major Features of LLMs

1. **Large**
    
    It refers to the large training dataset required to train the model. It also corresponds to large number of parameters which define the skill of solving. These parameters are the memories and the knowledge learnt by the model.
    
2. **General Purpose**
    
    It is sufficient to solve common problems. This is because of the commonality of human language. Since these models require lots of resources to train, only certain organizations have the capacity to create foundational models.
    
3. **Pretrained and Fine-tuned**
    
    The models are pretrained on large databases and then can be fine-tuned for a domain specific task using a small database.
    

## Types of LLMs

There are three types of LLMs and each of them require different type of prompting. Also, the first two types confuse easily so we need to use them carefully.

1. **Generic (Raw) Language Model**
    
    It predicts next token based on training data. It is just like autocomplete in search.
    
2. **Instruction Tuned Language Model**
    
    It predicts response to instructions given. Example: Summarize text, generate poem in given style, give synonyms, sentiment classification.
    
3. **Dialog Tuned Language Model**
    
    It is trained to have dialog to predict next response. It is a special case of instruction tuned where requests are questions. It is further specialization which is expected to have longer context and work better with natural question like phrasing.
    

### Chain of Thought Reasoning

Language models output better answers when they first output reason for the answer, rather than directly arriving to it. This is more prominent in numerical calculations.

## Benefits

1. **Single model for different tasks**
    
    Built using petabytes of data and billions of parameters. Can perform operations such as language translation, sentence completion, QA, etc.
    
2. **Fine-tuning requires minimal field data**
    
    These models have decent performance with little domain data. They can be used in few-shot or zero-shot scenarios.
    
3. **Continuous performance growth**
    
    These models can be continuously improved by providing more data and increasing the number of parameters.
    

## Example - PaLM

Pathways Language Model (PaLM) was released by Google in April 2022. It has 540 billion parameters and achieved state of the art performance on variety of tasks. It is a dense Decoder only transformer model.  
It leverages Google's new **Pathways system** that efficiently trains a single model across multiple TPU V4 pods. It is a new AI architecture that can handle multiple tasks at once, learn new tasks quickly and reflect a better understanding of the world. It enables PaLM to orchestrate distributed computation for accelerators.

## LLM Development vs Traditional Development

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701527080342/d9c75dc3-e2a8-4b7a-8c8e-86a68517955a.png align="center")

# LLM Use Cases

The course discusses Text Generation - Question Answering as an example application of LLMs.

## Question Answering (QA)

1. It is a subfield of Natural Language Processing. It answers questions posed in Natural Language.
    
2. QA systems are trained on large amount of text and code.
    
3. It is used for wide range of questions such as factual, definitional and opinion based.
    
4. ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701527308655/ed4aa19f-78b2-4c75-9cbb-b13e78f7f813.png align="center")
    

### **Generative QA**

Generates free text using context. It leverages text generation models and do not need domain knowledge.

### **Bard QA**

Performs operation as directed by the prompt given and also provides definition. Getting desired results require prompt design.

![Bard QA example 1 from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701527466585/50af77da-143f-493c-b581-7aa40b96211d.png align="center")

![Bard QA example 2 from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701527558122/26f352f6-cae6-41a6-9d74-36dab224fb87.png align="center")

# **Prompt Tuning**

Both prompt design and prompt engineering involve creating a prompt that is clear, concise and informative. But their differences are as follow:

## Prompt Design

1. Involves creating a prompt tailored for a specific task.
    
2. Requires instructions and context.
    
3. It is essential.
    

## Prompt Engineering

1. Create prompt to improve performance.
    
2. Requires domain knowledge as well as provided examples.
    
3. Involves using effective keywords.
    
4. Necessary for systems that require high accuracy or performance.
    

# Tuning LLMs

Adapting a Large Language Model to new domain is called Tuning.  
Example: Tuning for legal or medical domains.

## Fine-tuning

Fine-tuning an LLM requires bringing own dataset and then retraining every weight in LLM. This involves a big training job and hosting one's own model.

![Fine-tuning for Healthcare data from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701527763532/cdb002b9-03e8-4bfc-9463-62be8f81b5a4.png align="center")

This process is expensive and usually not realistic as every organization does not have sufficient resources to perform these tasks. Instead, Parameter Efficient Tuning Method (PETM) is employed.

## Parameter Efficient Tuning Method (PETM)

This involves tuning LLM without duplicating the model. The base model remains unaltered. Instead, small add-on layers are added and tuned, which can be swapped at inference time.  
Another form of simple PETM is prompt tuning which can also alter the model output without retraining it.

# Generative AI Development Tools

Discussion about Vertex AI Search and Conversation, Gen AI Studio and MakerSuite were covered in the last article: I[ntroduction to Generative AI (Part 2)](https://keoliya.hashnode.dev/intro-to-gen-ai-2)

# Ending Note

More details about prompts and prompt engineering will be discussed in subsequent posts. Stay tuned for lots of Generative AI content.