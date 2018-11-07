
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

