
# Summary
An end-to-end method for generating imgs from scene graphs.  
* explicitly reasoning about objects and their relationships.  
* method: 
```
1. uses graph convolution to process input graphs.  
2. layout <- predicting bounding boxes and segmentation masks for objects.  
3. layout -> an image with a cascaded refinement network.  
```
  
* train the network!  
trained adversarially against a pair of discriminators to ensure realistic outputs.  
* validation
on Visual Genome and COCO-Stuff(qualitative results, ablations and user studies).  
  
# Introduction
If out computer vision systems are to truly understand the visual world, they must be able not only recognize images but also to generate them.  
  
## why this task?
* previous method: text2image
```
RNN + GAN : natural language descriptions -> images.  
```
* problem
struggle with complex sentences containing many objects.  

* why is this problem?
complex sentence's information -- better --> a scene graph of objects and their relationships.  

* scene graphs have been used in:  
  1. semantic image retrieval.  
  2. evaluating image captioning.  
  3. improving image captioning.  
  
* sentences/images -> scene graphs: available now!

## challenges
1. develop a method for processing scene graph inputs.  
    * a graph convolution network which passes information along graph edges
  
2. bridge the symbolic graph-structured input and the two-dimensional image output.  
    * a scene layout <- predicting bounding boxes and segmentation masks for all objects in the graph.  
    * an image <- scene layout by a CRN(cascaded refinement network)
  
3. ensure that the generated images are realistic and contain recognizable objects.  
    * train adversarially against a pair of discriminator networks operating on image patches and generated objects.  

**All components of the model are learned jointly in an end-to-end manner!**

##  Experiments
Visual Genome: human annotated scene graphs.   
COCO-Stuff.  
User Studies.  

## Method
image generation network f: scene graphs -> images.  
![](https://latex.codecogs.com/gif.latex?%5Cwidehat%7BI%7D%3Df%28G%2Cz%29)  
G - scene graph, z - noise.  
  
graph convolution network: G -> embedding vectors for each object.    

# Related Work(under Construction)
## Generative Image Models
three categories:  
1. GANs jointly learn a generator for synthesizing images and a discriminator classifying images as real or fake.  
2. Variational Autoencoders use variational inference to jointly learn an encoder and decoder mapping between images and latent codes.  
3. autoregressive approaches model likelihoods by conditioning each pixel on all previous pixels.  
  
## Conditional Image Synthesis
Conditional Image Synthesis conditions generation on additional input.  
GANs can be conditioned on category labels by:  
1. providing labels as an additional input to both generator and discriminator
2. forcing the discriminator to predict the label.**(We are using this one)**  

## Scene Graphs
## Deep Learning on Graphs
