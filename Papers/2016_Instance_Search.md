
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

#### Improvement to R-CNN: SPP-net
* used a Spatial Pyramid based pooling layer to improve classification and detection performance.  
* significantly decreased computational time by pooling region features from conv features instead of forward passing each region crop through all layers in the CNN
#### Improvement to R-CNN: Fast R-CNN
* used the speed strategy as SPP-net
* replaced the post-hoc training of SVM classifiers and box regressors with an end-to-end training solution
#### Improvement to R-CNN: Faster R-CNN
* removed the object proposal dependency by introducing a Region Proposal Network(RPN)
* RPN shares features with the object detection network to simultaneously learn prominent object proposals and their associated class probabilities

#### In this work
* take advantage of the end-to-end self-contained object detection architecture of Faster R-CNN
* extract both image and region features for instance search

## Methodology

### CNN-based Representations
This paper: features from object detection CNN -> instance search

#### Faster R-CNN
Two branches that share conv layers:  
* a Region Proposal Network: learns a set of window locations
* a classifier: learns to label each window as one of the classes in the training set

#### Image-wise pooling of activations(IPA)=image-wise descripters
* IPA: The activations of a conv layer -> a global image descriptor of the same dimension as the number of filters in the conv layer
* max/sum pooling strategies are compared 

#### Region-wise pooling of activations(RPA)=region-wise descriptors
* Faster R-CNN's region pooling layer: extracts the conv activations for each of the object proposals learned by the RPN
* give raise to the region-wise descriptors
* max/sum pooling strategies are compared

#### max pooling
max-pooled features:  
1. l_2-normalizing.  
  
#### sum pooling
sum-pooled features:  
1. l_2-normalizing.  
2. whitening.  
3. l_2-normalizing.  

### Fine-tuning Faster R-CNN
#### Why fine-tune?
Want a fine-tuned Faster R-CNN to:  
* obtain better feature representations for image retrieval
* improve the performance of spatial analysis and reranking

#### What to do?
Fine tune to detect the query obejcts to be retrieved by our system.  

#### How did we do this?
Modify the architecture of Faster R-CNN to output the regressed bounding box coordinates and the class scores for each one of the query instances of the tested datasets.  

#### Two modalities of fine-tuning
* Fine-tuning Strategy #1:  
Only the weights of the fc layers in the classification branch are updated. 
* Fine-tuning Strategy #2:  
Weights of all layers after the first two conv layers are updated.  


### Image Retrieval
The three stages of the proposed instance retrieval pipeline.  
#### 1. Filtering Stage
* In this stage, the whole image is considered as the query
* IPA: build image descriptors for both query and database images
* At test time, the descriptor of the query image is compared to all elements in the database, which are then ranked based on the cosine similarity.  

#### 2. Spatial Reranking
The top N elements are locally analyzed and reranked.  

#### Class-Agnostic Spatial Reranking(CA-SR)
For every image in the top N ranking, the RPA for all RPN proposals are compared to the region descriptor of the query bounding box.  
* Obtain the region-wise descriptor of the query object:  
Warp its bounding box to the size of the feature maps in the last conv layer and pool the activations within this area.  
* The region with the maximum cosine similarity for every image in the N images gives the object localization, and its score is kept for ranking.  

#### Class-Specific Spatial Reranking(CS-SR)
Using a network that has been fine-tuned with the instances to retrieve.  
* use the direct classificatoin scores for each RPN proposal as the similarity score to the query object.  
* the region with maximum score is kept for visualization, used to rank the image list.  

#### 3. Query Expasion(QE)
The image descriptors of the top M elements of the ranking are averaged together with the query descriptor to perform a new search.  

## Experiments

### Datasets
Three datasets to assess the methodologies:
#### Oxford Buildings
* 5063 images
* 55 query images of 11 different buildings in Oxford
* 5 images/instance
* a bounding box surrounding the target object is provided for query images
#### Paris Buildings
* 6142 images of Paris
* 55 query images of 11 buildings with associated bounding box annotations
#### INS 2013
* a subset of 23,614 keyframes from TRECVid Instance Search(INS) dataset
* only those keyframes that are relevant for at least one of the queries of INS 2013

### Experimental Setup
#### architecture choice for Faster R-CNN
Tried both:  
* VGG16
* ZF
#### build the image descriptors
Use he last conv layer to build:
* conv5 for ZF(descriptor dimension:256)
* conv5.3 for VGG16(descriptor dimension:512)

#### image rescaling
* shortest side is 600 pixels

#### GPU setting
* Nvidia Titan GPU


### Off-the-shelf Faster R-CNN features
#### Compare the sum and max pooling of image- and region-wise descriptors
* filtering stage: sum pooling > max pooling -> **set IPA-sum descriptors**
* reranking stage: sum pooling < max pooling -> **set RPA-max descriptors**

#### Special: spatial reranking not working in INS13!
* Qualitativey evaluate the two pooling strategies:  
Top rankings of INS13 queries, spatially reranked with region-wise max and sum poolied descriptors.  
* although mAP is similar, the object location  obtained with max pooling is more accurate!  

#### Network Choice
* ZF and VGG16(Faster R-CNN) trained on Pascal VOC and COCO
* features pooled from the deeper VGG16 is better

#### Effect of QE
* QE: if applied, M=5
* QE is significant! Reasonable, cuz spatial reranking worked in two Dset!
* QE is useful following spatial reranking even in INS13! -> the images that fall on the very top of the ranking are more useful to expand the query than the ones in the top of the first ranking.  


### Fine-tuning Faster R-CNN

#### Network Choice
VGG16 Faster R-CNN model, pretained with the objects of the Microsoft COCO dataset.  

#### Modification to the Network
modify the output layers in the network to return:  
* 12 class probabilities:  
1. 11 buildings in the dataset.  
2. an extra class for the background.  
* corresponding regressed bounding box coordinates

#### Training data for Oxford and Paris: 11*5*2=110 images
* 5 images provided for each one of the building and their bounding box locations
* augment the set by horizontal flip

#### Training data for INS13: 30*4*2=240 images
* 30 different query instances
* 4 images each
* horizontal flip
* output class: 30+1(background)

#### iteration of fine-tuning: 5000
* considering the small number of training examples

#### training
* approximate joint training strategy: trains the RPN and classifier branches at the same time by multi-task loss
* train a separate network for each of the tested datasets using different fine-tuning modalities
* time: 30, 45m for each strategy on Nvidia Titan X GPU

#### finetuning strategies #1
* only update parameters in fc layers.  
* In INS13, performance did not improved after using CS-SR: only fine-tuning fc layers not sufficient to detect

#### finetuning strategies #2
* update params in last conv layer, fc layers, RPN proposals
* After fine-tuning, more neurons in the conv layer reacts to query objects
* fine-tuned features are already discriminant enough to retrieve objects in 2 datasets, but bad in INS3. 
* complicated datasets benefit most from fine-tuned features and spatial reranking 

#### ft#2>ft#1
ft#2: feature and RPN layers to adapt to the query objects

### Comparison with state-of-the-art
#### Why somebody is doing better without reranking/qe?
* the difference in CNN architecture: Faster R-CNN vs VGG16
* training data: Pascal VOC vs ImageNet
* input image size: 600px wide vs full resolution

#### results obtained with fine-tuned features(strategy 2) is competitive!

