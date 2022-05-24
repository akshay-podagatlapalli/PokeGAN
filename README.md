# PokeGAN - A Generative Adversarial Network to Create New Pokemon

![whos the pokemon](https://user-images.githubusercontent.com/65557678/169938561-99d5eb43-f808-4d30-baba-ed7af206dcb1.png)


## Project Description and Motivation
After thinking long and hard (a quick google search) for new ML project ideas, I found the use of GANs (Generative Adversarial Networks) 
to be the most coolest and the one with a greater freedom of degree in what I could do. Specifically, I wanted to generate new Pokemon with 
this model. While there are earlier iterations on this project that I have taken inspiration from, such as [this](https://blog.jovian.ai/pokegan-generating-fake-pokemon-with-a-generative-adversarial-network-f540db81548d) and [this](https://www.kaggle.com/code/shunsukeozeki/pokemon-dcgan-with-pytorch-create-new-pokemon), my goal with this project was to learn more about GANs rather than generate new Pokemon. My expectations about this project was not necessarily results oriented at the start and any meaningful synthesis of new Pokemon was treated as a welcomed surprise since fine-tuning a GAN is rather tricky.    

## Data Used
The data for this project was obtained from [here](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)

## Code
Pytorch is the primary module used in the development of this model. 

The model used here is the Deep Convolutional Generative Adversarial Network otherwsie known as [**DCGAN**](https://arxiv.org/pdf/1511.06434v2.pdf)
![DCGAN Architecture](https://user-images.githubusercontent.com/65557678/169937851-c685ff64-92ee-4ec5-bf3e-5a3a1c2e1483.png)

- Refer to pokedatasetloader.py to see how the data was pre-processed, transformed using various data augmentation functions, and a neat function 
  I also included, that I painstakingly researched to calculate the mean and standard deviation of any image dataset. 
  
- Refer to pokeGAN.py to see the implementation of the GAN model. The model was developed to generate 64px * 64px size images. 

- Refer to main.py to view the training of the model. Based on the current hyperparameter settings, the model is yet to perform much better. This is clearly     evident from the loss plot below. However, the results visualized during the training from video below demonstrate some "blobs" confined within some legible   outlines. The details within the outlines however, have to be improved further. 

![loss_func](https://user-images.githubusercontent.com/65557678/169939034-5f668e6b-1f5f-4139-99ab-fec51e63af17.jpg)

https://youtu.be/57Vkrnna4CU
