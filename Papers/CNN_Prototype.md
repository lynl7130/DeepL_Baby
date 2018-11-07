
# ImageNet Classification with Deep Convolutional Neural Networks

[NIPS 2012 paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

## Summary
* NN:  
60,000,000 parameters and 650,000 neurons.  
5 conv layers(some followed by max-pooling layers).  
3 fc layers + final 1000-way softmax layer.  
* facilitate training:  
non-saturating neurons.  
efficient GPU implementation of the conv operations.  
* reduce overfitting:  
dropout.  

## Introduction
### Why CNN is good?
Datasets will never be large enough!  
Our model need prior knowledge to compensate for this.  
CNNs make strong and mostly correct assumptions about the natural of images:  
* stationarity of statistics
* locality of pixel dependencies
-> much easier to train than feedforward NNs, equivalent or better performance.  

### Why CNN is good for now?
GPU is here!

**Faster GPUs and bigger datasets would lead to better models!**  

## The Dataset
ImageNet consists of variable-resolution images.  
-> down-sampled the images to 256x256:  
1. rescaled the image such that the shorter side was of length 256.  
2. cropped out the central 256x256 patch from the resulting image.  
-> subtracting the mean activity over the training set from each pixel.  

## The Architecture
### [Overall Architecture](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf#page=5)

### ReLU Nonlinearity
Non-saturating nonlinearities are much faster to train than saturating nonlinearities. 
#### What is saturating?
f is non-saturating iff (|limz→−∞f(z)|=+∞)∨|limz→+∞f(z)|=+∞).  
f is saturating iff f is not non-saturating.  

### Training on Multiple GPUs
#### How to the break the limitation of GPU?
Spread the net across 2 GPUs.  
* GPUs communicate only in certain layers.  

### Local Response Normalization
Although ReLU do not require input normalization to prevent saturating, this normalization helps.  
![](https://latex.codecogs.com/gif.latex?a_%7Bx%2Cy%7D%5Ei): the activity of a neuron computed by applying kernel i at position (x,y).  
#### Response-Normalized activity: 
![](https://latex.codecogs.com/gif.latex?b_%7Bx%2Cy%7D%5Ei%20%3D%20%5Cfrac%7Ba_%7Bx%2Cy%7D%5Ei%7D%7B%28k%20&plus;%20%5Calpha%20%5Csum_%7Bj%20%3D%20max%280%2C%20i-n/2%29%7D%5E%7Bmin%28N-1%2C%20i&plus;n/2%29%7D%7B%28a_%7Bx%2Cy%7D%5Ej%29%5E2%7D%29%5E%5Cbeta%7D)  
![](https://latex.codecogs.com/gif.latex?k%2Cn%2C%5Calpha%2C%5Cbeta): hyperparameters.  

### Overlapping Pooling
overlapping pooling makes it slightly more difficult to overfit.  

## Reducing Overfitting
### Data Augmentation
computationally free label-preserving transformations.  
#### Method 1
generating image translations and horizontal reflections.  
#### Method 2
altering the intensities of RGB channels.  

### Dropout
a very efficient version of model combination.  
Everytime an input is presented, the NN samples a different architecture.  
* Dropout roughly doubles the number of iterations required to converge.  

## Discussion
Ultimately we would like to use very large and deep convolutional nets on videosequences where the temporal structure provides very helpful information that is missing or far less obvious in static images.
