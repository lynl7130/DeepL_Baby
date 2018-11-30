
# Faster R-CNN Features for Instance Search

## Summary
* different strategies to make use of CNN features from an object detection CNN
* provides a baseline: use Faster R-CNN features to describe images and sub-parts
* instance search pipeline:  
1. filtering stage.  
2. spatial reranking.  
* investigated the suitability of Faster R-CNN features when NN is fine-tuned for certain objects
* Datasets: Oxford Buildings 5k, Paris Buildings 6k and a subset of TRECVid Instance Search 2013

## Introduction

### What is instance search?
The task of retrieving the images from a database that contain an instance of query

### Why CNN?
* have achieved state of the art performace in:  
1. image classification.  
2. object detection.  
3. semantic segmentation.  
* CNN learns generic feature representations that could be used to solve tasks for which they had not been trained.  
* CNN(pretrained for image classification) has improved image retrieval performance!

### Instance search systems
#### steps:
1. fast first filtering  
2. reranking  

#### fast first filtering stage?
* in this stage, all images in a database are ranked according to their similarity to the query
* **we need this** because more computationally expensive mechanisms would be applied to top items

#### reranking strategies
* strategy choices:  
1: geometric verification  
2: spatial analysis  
* often followed with query expansion(pseudo-relevance feedback)

#### Spatial reranking?
1. usage of sliding windows at differenct scales and aspect ratios over an image.  
2. each window is then compared to the query instance in order to find the optimal location that contains the query.  
* step 2 requires the computation of a visual descriptor on each of the considered window
* Spatial reranking strategy ~ object detection algorithm!

### Hint of this paper
* such object detection algorithm has become deprecated!
* Object Detection CNNs are trained in an end-to-end manner to simultaneously learn obejct locations and labels.  

### What we did?
Explores the suitability of off-the-shelf and fine-tuned features from an object detection CNN for the task of instance retrieval.  
#### Three contributions
* propose to use a CNN pre-trained for object detection to extract convolutional features both at global and local scale in a single forward pass of the image throught the network
* explore simple spatial reranking strategies, which take advantage of the locations learned by a Region Proposal Network(RPN) to provide a rough object localization for the top retrieved images of the ranking.  
* analyze the impact of fine-tuning an object detection CNN for the same instances one wants to query in the futrue. We find such a strategy to be suitable for learning better image representations.  


## Related Work
### CNNs for Instance Search
#### Early works  
Use features from pre-trained image classification CNNs for instance search.  
* demonstrated the suitability of features from fc layers for image retrieval
* someone improved the result by combining fc layers extracted from different image sub-patches

#### Second generation
Explore the usage of other layers in the pretrained CNN 
* found that conv layers >> fc layers at image retrieval tasks
* someone proposed a compact descriptor composed of the sum of the activations of each of the filter responses in a conv layer
* someone introduced R-MAC: a compact descriptor composed of the aggregation of multiple region features.  
* someone improved by applying non-parametric spatial and channel-wise weighting strategies to the conv layers

#### In this work
* similar to former: use conv features of a pretrained CNN
* use a state-of-art object detection CNN to extract both image- and region-based conv features in a single forward pass


### Object Detection CNNs
#### R-CNN
instead of full images, the regions of an object proposal alg are used as inputs to the network.  
At test time, fc layers for all windows were extracted and used to train a bounding box regressor and classifier.  
* someone proposed SPP-net:  
1. used a Spatial Pyramid based pooling layer to improve classification and detection performance.  
2. significantly decreased computational time by pooling region features from conv features instead of forward passing each region crop through all layers in the CNN
* 
