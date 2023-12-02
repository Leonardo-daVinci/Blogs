---
title: "Introduction to Generative AI (Part 2)"
seoTitle: "Introduction to Generative AI"
seoDescription: "Basic introduction to Generative Artificial Intelligence from Google Cloud course."
datePublished: Fri Dec 01 2023 15:00:13 GMT+0000 (Coordinated Universal Time)
cuid: clpmr4cul000809jscy980chc
slug: intro-to-gen-ai-2
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1701368539736/6f8be907-dc3c-4b03-9c01-a4422708b452.png
tags: artificial-intelligence, machine-learning, google-cloud, google, generative-art, transformers, bard, vertex-ai

---

This is collection of notes from the [**Generative AI course**](https://www.cloudskillsboost.google/course_templates/536) on Google Cloud taught by Dr. Gwendolyn Striping. Some images are taken from the course itself.

It is a detailed compilation and annotated excerpts will be available on my [LinkedIn profile](https://www.linkedin.com/in/akshit-keoliya/).

The following is a continuation of the [Introduction to Generative AI (Part 1) article](https://keoliya.hashnode.dev/introduction-to-generative-ai-part-1). Make sure to go over it before continuing below.

# Course Overview

The course is divided into four parts. We will go through the last 2 ones in this article.

1. Defining Generative AI
    
2. Working of Generative AI
    
3. Generative AI Model Types
    
4. Generative Applications.
    

# Model Types

Generative models can be divided into 4 types as follows:

1. **text-to-text**  
    These models learn mapping between pair of texts.
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701365711715/cf73d683-e98d-40ff-9734-26d7f851604f.png align="center")
    
2. **text-to-image**  
    These models are trained on images with short text description.
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701365727866/c40e29c7-372c-4429-bd54-eb01f6a71295.png align="center")
    
3. **text-to-video and text-to-3D**  
    These models generate video representation from input text which can be a sentence or full script. We can also generate 3D models based on text description.
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701365743479/09c1f542-e7fb-4f76-8e52-03027ad8a408.png align="center")
    
4. **text-to-task**  
    These models are trained to perform specific tasks such as Question Answering, Search or Prediction based on text input.
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701365760239/965d0ad1-5546-4752-8fa4-680bbe2b5d68.png align="center")
    

# Applications

![Generative AI Application Landscape image from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701367093945/c765ba48-1d71-452a-bd29-55c9f0ab60c7.png align="center")

The course gives example application of **Code Generation** using Bard. We can give a prompt to Bard, get results and also export them to Google Colab.  
Bard can also provide number of other functionalities such as Debugging Code, Code Explanation, Code Translation and Documentation.

## Foundation Models

Foundation Models are Large AI models that are trained on vast quantities of data. These are adapted and fine-tuned to perform downstream tasks such as Sentiment Analysis, Image Captioning and Object Recognition.

![Foundation Models Image taken from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701366912680/240f38ef-3965-44f6-8fa0-692e0ca6eede.png align="center")

Google Cloud's Vertex AI provides model garden that consists of various foundation models for variety of use cases. Examples of such foundational models include **PaLM** (Pathways Language Model) for Chat and Text, Stable Diffusion and CLIP.

![Model Garden Image from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701366959811/b836b7fe-f716-48d0-8e22-ccaa3bb55d72.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701366998854/6c582a34-cee2-4402-b7c4-665d296e74d7.png align="center")

## Google's Offerings

### Generative AI studio

Generative AI studio lets you quickly explore and customize Gen AI models that you can leverage in Google Cloud. Other functionalities provided by GenAI Studio are as follows:

1. Fine tune models
    
2. Deploy Models to Production.
    
3. Create Chatbots
    
4. Image Generation.
    
5. Community Forum.
    

### Vertex AI Search and Conversation

1. Build Generative AI Applications with little-to-no coding or ML experience.
    
2. Utilize Drag-and-drop interface.
    
3. Visual Editor that can create and edit application content.
    
4. Built in Conversational AI engine to help users interact with the app using Natural Language.
    
5. Create Digital Assistants, chatbots, custom search engines, knowledge bases.
    

### PaLM API & MakerSuite

PaLM API can be utilized to test, experiment and prototype generative applications using Google's LLMs and GenAI tools. PaLM API is integrated into MakerSuite.  
MakerSuite can be used to access APIs in graphical user interface, and contains various tools such as follows:

1. **Tools for Model training**  
    We can utilize different types of algorithms and check which suits our dataset or use case better.
    
2. **Tools for Model Deployment**  
    We can deploy our model with variety of different options.
    
3. **Tools for Model Monitoring**  
    We can track our model's performance using a dedicated dashboard. We can also use different metrics to evaluate our model performance.
    

# Ending Note

The course provides an extensive list of documents that can help enhance our understanding. You can follow the link [here](https://drive.google.com/file/d/1k35iYhxkkcp_bQa3X4bKNkpZMoMiMWMA/view?usp=sharing) to access these documents.

We will cover **Introduction to Large Language Models** course next in the learning path. More details about Transformer Models, Stable Diffusion and Attention Mechanism will be discussed in subsequent posts. Stay tuned for lots of Generative AI content!