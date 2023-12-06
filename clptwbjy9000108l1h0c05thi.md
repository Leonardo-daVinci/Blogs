---
title: "Introduction to Image Generation"
seoTitle: "Introduction to Image Generation"
seoDescription: "Basics of Image Generation using Diffusion Models. How Diffusion process works and how to use it to generate novel images."
datePublished: Wed Dec 06 2023 15:00:10 GMT+0000 (Coordinated Universal Time)
cuid: clptwbjy9000108l1h0c05thi
slug: intro-to-image-gen
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1701871712406/c94e3f70-d9c3-4379-a61d-3e25733e5ecc.png
tags: artificial-intelligence, machine-learning, google-cloud, google, image-generation, generative-ai, vertex-ai, diffusion-models

---

This is a collection of notes from the [Introduction to Image Generation course](https://www.cloudskillsboost.google/paths/183/course_templates/541) on Google Cloud taught by **Kyle Steckler**. Some images are taken from the course itself.

It is a detailed compilation and annotated excerpts will be available on my [**LinkedIn profile**](https://www.linkedin.com/in/akshit-keoliya/).

# Course Overview

1. Image Generation Families
    
2. Diffusion models
    
3. Working of Diffusion Models
    
4. Recent Advancements
    

# Image Generation Families

1. **Variational Autoencoders (VAEs)**
    
    Encode images to a compressed size, then decode back to original size, while learning the data distribution.
    
2. **Generative Adversarial Models (GANs)**
    
    Pit two neural networks against each other. One model (Generator) creates candidates while the second one (Discriminator) predicts if the image is fake or not. Over time Discriminator gets better at recognizing fake images and Generator gets better are creating real images.
    
3. **Autoregressive Models**
    
    Generates images by treating an image as a sequence of pixels. Draws inspiration from how LLMs handle text.
    

# Diffusion Models

Diffusion Models draw inspiration from Physics, especially **Thermodynamics**. Their usability has seen a massive increase in research space and now commercial spaces too. They underpin many state-of-the-art models that are famous today such as **Stable Diffusion**.

## Types of Diffusion Models

1. ![Types of Diffusion models image taken from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701868876201/caebdd5d-9ff5-4282-8e0c-f3b913aa2fca.png align="center")
    
    **Unconditioned Generation**
    
    Models have no additional input or instruction. Can be trained from images of a specific thing for creating images such as human faces or enhancing image resolutions.
    
2. **Conditioned Generation**
    
    These models can generate images using a text prompt or edit the image itself using text.
    

# Working of Diffusion Models

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701869425718/b3f70986-3845-4b5f-9393-53988a7388e6.png align="center")

The process consists of two stages:

1. **Forward Diffusion**
    
    Systematically and slowly destroy the structure in a data distribution. This is done by adding Gaussian noise iteratively to the existing image.
    
2. **Reverse Diffusion**
    
    Restore structure in data yielding a highly flexible and tractable generative model of data. The model learns how to de-noise an image which can help generate novel images.
    

## Denoising Diffusion Probabilistic Models (DDPM)

The goal is to make a model learn how to de-noise or remove noise from an image.  
Then, we can start from pure noise and then iteratively remove noise to synthesize a new image.

### Steps:

1. We start with a large dataset of images.
    
2. **Forward Diffusion Process**
    
    For each image, we add a little bit of Gaussian noise at each timestep.
    
3. **Iteration through T=1000 timesteps**
    
    The above process is repeated for **T** timesteps, adding more noise iteratively to the image from the last timestep.
    
4. **End of Forward Diffusion**  
    Ideally, by the end of the forward diffusion process, all structure in the image is gone and we should have pure noise.
    
5. **Reverse Diffusion Process**  
    To go from a noisy image to a less noisy one, we need to learn how to remove the Gaussian noise added at each timestep.
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1701870107190/6478c679-3552-4e35-8648-59d73ad74623.png align="center")
    
6. **Denoising Model**  
    We train a machine learning model that takes in noisy images as input and predicts the noise that's been added to it.
    
7. **Training Denoising Model**  
    The output of the Denoising model is predicted noise and we know what noise was initially added. We can compare them and thus train the model to minimize the difference between them. Over time the model gets very good at removing noise from images.
    
    ![DDPM Training image from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701870200647/c57981c1-57e8-4731-bca1-509b25ef0599.png align="center")
    
8. **Image Generation**  
    For generating images, we start from pure random noise and pass it through our Denoising model multiple times to generate new images.
    
    ![Image Generation image from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701870278174/4a77b07c-a557-47d9-88f0-882c5124468e.png align="center")
    

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">The model is able to learn the data distribution that it has seen and then sample from that distribution to create new novel images.</div>
</div>

# Recent Advancements

Lots of advancements have been made to generate images faster and with more control.  
By combining the power of LLMs and Diffusion Models, there has been a huge success in creating context-aware, photorealistic images using text prompts.  
Example: Imagen on Vertex AI

![Recent Advancements image from Google Cloud](https://cdn.hashnode.com/res/hashnode/image/upload/v1701870368940/ae15642d-7a06-46e6-a374-fe805d752fe4.png align="center")

# Ending Note

We learned how diffusion models have transformed the image generation space and continue to be at the core of modern image generation models.  
Next, we will learn about Encoder-Decoder Architecture which serves as the building blocks of all Transformer Models we use today.

Stay tuned for lots of Generative AI content!