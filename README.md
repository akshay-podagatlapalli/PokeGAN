# PokeGAN - A Generative Adversarial Network to Create New Pokemon

![whos the pokemon](https://user-images.githubusercontent.com/65557678/169938561-99d5eb43-f808-4d30-baba-ed7af206dcb1.png)


<h2>Introduction</h2>>
<p>Generative Adversarial Networks (GANs) are a type of unsupervised neural networks that falls under the purview of deep learning models. They are commonly used in the image-processing domain to create art <b>[1]</b>, music <b>[2]</b>, or to improve the quality of low-resolution images/videos <b>[3]</b>. Recently, researchers at the University of Toronto used their applications in biochemistry and medical studies to generate 30,000 designs for six different new compounds that were found to imitate drug-like properties and target a protein involved in fibrosis <b>[4]</b>. I trained a GAN model to generate fake Pokémon.</p>

<p>Because GANs are primarily taught to learn the distribution of any given dataset, the applications are really domain-independent. GANs will be able to replicate aspects of our environment given a well-defined dataset. The key constraint is the computing power required to train these models, which is further hampered by the fact that they are notoriously difficult to train, necessitating extra training time and computational power.</p>

<p>What makes training these models so difficult? Understanding this requires taking a look under the hood of this model. It was first proposed in the landmark paper Generative Adversarial Nets <b>[5]</b>, and it presents a paradigm in which two fully-connected neural networks (NN) compete in a zero-sum game. One of the NNs, known as the generative network or the generator, will work to generate "false" data, while the other, known as the discriminative network or the discriminator, will work to evaluate and distinguish between the actual and fake data.</p>

![fig1](https://github.com/akshay-podagatlapalli/PokeGAN/assets/65557678/787790e9-e4b2-4e5e-8174-44516839bd2c)

<p>Figure 1 depicts an abstract representation of how the generator (green line) would "dupe" the discriminator. The generator will train until its distribution resembles that of the real dataset (black dotted line). Given that the generator's job is to trick the discriminator until it can no longer distinguish between the two distributions, the discriminative distribution (blue dashed line) should flatten when the fake and actual distributions become indistinguishable.</p>

<p>The model used to create the "fake" Pokémon in this case is known as the DCGAN, which stands for Deep Convolutional Generative Adversarial Network. This model, unlike the fully connected models suggested in <b>[5]</b> employs two convolutional neural networks (CNNs) for the generative and discriminative networks.
The discriminator is a CNN model, whereas the generator is a deconvolutional neural network, which works inversely to a conventional CNN model. Where a CNN learns the spatial hierarchies of features within an image, moving from granular to high level details, the deconvolutional neural network or the generator learns to convert the latent space inputs into an actual image, generating meaning from noise, by regularly updating its weights by learning how the discriminator evaluates the images fed into its network. This is depicted in the figure below, which shows how data flows through a generative neural network.</p>


<img width="685" alt="fig2" src="https://github.com/akshay-podagatlapalli/PokeGAN/assets/65557678/359474b3-1e67-4faa-bae6-ce82912d4a13">

<p>By providing a random seed, the generator begins to produce candidates for the discriminator from a latent space and maps it to the distribution of the dataset being used. A latent space is a representation of compressed data best explained in <b>[7]</b>. The space is initially populated randomly, but as the generator begins to understand a dataset’s distribution, the latent space would slowly start to be populated by features learned from the distribution. In contrast, the discriminator is trained on random datapoints drawn from the actual dataset. Both models will be trained until they achieve an acceptable level of accuracy, with each model undergoing backpropagation individually to enhance accuracy.</p>

<p>This is further emphasized in Figure 3, where we see how the data produced by the generator is fed to the discriminator along with the real data.</p>

![fig3](https://github.com/akshay-podagatlapalli/PokeGAN/assets/65557678/875228c1-47e9-4106-92c5-eb3843eef1b9)


<h2 class="section-heading">Data Collection and Processing</h2>

<p>For this project, the Pokémon dataset was acquired via Kaggle. The original dataset is made up of 819 photos that were uploaded as .png files with a resolution of 256x256 pixels <b>[9]</b>. Because GANs are notoriously data hungry <b>[10]</b>, the size of this dataset was expanded 13 times prior to training by executing a data augmentation step.</p>

![fig4](https://github.com/akshay-podagatlapalli/PokeGAN/assets/65557678/99ac6394-9617-4b74-bbbf-6846b33f6371)

<p>Despite expanding the dataset size, the results appeared to follow the same patterns as those seen  on Kaggle <b>[9]</b>. Normalizing the dataset by calculating the mean and standard deviation did not appear to improve the results and actually worsened them.</p>

<p>A few considerations regarding the hyperparameters were made before training the model. The training was repeated four times to see if altering certain hyperparameters affected the quality of the results. The <b>learning rate</b>, <b>batch size</b>, <b>latent space</b>, and <b>number of epochs</b> were all altered. Because of memory constraints caused by the number of layers in the models, the 256x256 input picture was scaled down to accept 64x64 and 128x128 images. The change in the input did not appear to drastically change the resolution of the outputs either.</p>

<p>Setting the batch size to a smaller value would prevent the discriminator from quickly outperforming the generator, leading to poorer results. The learning rate was also set to a conservative value; 1e-4 as it led to better results, purely based on the results observed over the 4 iterations. The latent space was primarily changed based on the assumption that because this value represented “compressed” data, the generator would reconstruct the distribution of the dataset from a larger value (or from larger latent space), ultimately leading to better results. Finally, the training time or epochs were chosen based on prior implementations of this project, other similar projects, and the constraints of my PC.</p>

<p>The values that were selected for each of the iterations are presented in the <b>Table 1</b> below</p>  


<table style="margin-left:auto;margin-right:auto;">
<colgroup>
<col width="15%" />
<col width="25%" />
<col width="20%" />
<col width="25%" />
<col width="30%" />
<col width="30%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align:center">Run</th>
<th style="text-align:center">Model Version</th>
<th style="text-align:center">Batch Size</th>
<th style="text-align:center">Learning Rate</th>
<th style="text-align:center">Latent Space</th>
<th style="text-align:center">Epochs</th>
</tr>
</thead>
<tbody>
<tr>
<td markdown="span" style="text-align:center">1</td>
<td markdown="span" style="text-align:center">64 px</td>
<td markdown="span" style="text-align:center">128</td>
<td markdown="span" style="text-align:center">1.00E-04</td>
<td markdown="span" style="text-align:center">64</td>
<td markdown="span" style="text-align:center">100</td>
</tr>
<tr>
<td markdown="span" style="text-align:center">2</td>
<td markdown="span" style="text-align:center">64 px</td>
<td markdown="span" style="text-align:center">64</td>
<td markdown="span" style="text-align:center">2.00E-04</td>
<td markdown="span" style="text-align:center">128</td>
<td markdown="span" style="text-align:center">70</td>
</tr>
<tr>
<td markdown="span" style="text-align:center">3</td>
<td markdown="span" style="text-align:center">128 px</td>
<td markdown="span" style="text-align:center">128</td>
<td markdown="span" style="text-align:center">2.00E-04</td>
<td markdown="span" style="text-align:center">256</td>
<td markdown="span" style="text-align:center">200</td>
</tr>
<tr>
<td markdown="span" style="text-align:center">4</td>
<td markdown="span" style="text-align:center">128 px</td>
<td markdown="span" style="text-align:center">64</td>
<td markdown="span" style="text-align:center">1.00E-04</td>
<td markdown="span" style="text-align:center">256</td>
<td markdown="span" style="text-align:center">100</td>
</tr>
</tbody>
</table>


<h2 class="section-heading">Results</h2>

<p>The results for each of the runs, presented in <b>Table 1</b> are presented below</p>


![fig5](https://github.com/akshay-podagatlapalli/PokeGAN/assets/65557678/9ca84f0c-1f65-4561-87e0-d85cd4dd6a5e)

![fig7](https://github.com/akshay-podagatlapalli/PokeGAN/assets/65557678/64c91dcb-6bce-42ad-9176-10be10df25f7)

![fig10](https://github.com/akshay-podagatlapalli/PokeGAN/assets/65557678/638e94b6-c56d-41ac-b185-dde72b0be038)


<p>The Pokémon generated using this model have distinct shapes and colours, but they lack features such as faces, limbs, or appendages such as tails, wings, horns, fins, and so on that are commonly seen on Pokémon. The losses for both models appear to raise concerns about mode collapse and/or failure of convergence based on the loss plot. When the generator's loss begins to oscillate repeatedly with the same <b><a href="https://machinelearningmastery.com/wp-content/uploads/2019/07/Line-Plots-of-Loss-and-Accuracy-for-a-Generative-Adversarial-Network-with-Mode-Collapse.png">oscillation loss pattern</a></b>, mode collapse might have occurred. It also results in very little diversity among the samples generated. However, the outcomes are far from identical. While it is evident that the loss functions for the generator and discriminator do not converge, it would also lead to the results simply producing plain noise as in Figure 7 below.</p>



<p>As a result, in addition to determining the best combination of hyperparameters, three additional runs were carried out to see whether similar patterns in the loss functions from Figure 6 maintained.</p>



<p>The loss functions for the generator and discriminator in Figures 9, 10, and 12 are observed to follow a general trend, seen in Figure 6. The results obtained throughout each of these runs are also identical to the ones shown in Figure 5. However, when examined through a forced creative lens, the results of the third run appear to show some form of limbs and appendages. Except for a couple in the last row of Figure 11, none of the results are truly legible. Because this model was trained for the longest time, 200 epochs, there is a significant likelihood that training the DCGAN model using the hyperparameters from RUN 3 for an even longer time will result in more defined outcomes.</p>

<p>Despite the slightly underwhelming results, I believe the project has great promise. Curating a more well-defined dataset with greater care to the data augmentation step could yield more clarity to the results presented. Furthermore, by labelling the data, and introducing noisy labels through the flipping the labels (by labelling real data as fake), the discriminator could be confused., thereby improving the results. In addition, I did not include many filters in the CNN models. Increasing the number of filters would also aid the CNN in extracting more information from images.</p>
