
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
2. 
