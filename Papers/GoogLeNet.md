
# Going deeper with convolutions

[CVPR 2015 paper](https://arxiv.org/pdf/1409.4842.pdf)

## Summary
* utilized computing resources inside the network:  
increasing the depth and width of the network while keeping the computational cost constant.  
* Hebbian principle
* multi-scale processing
* approximating the expected optimal sparse structure by readily available dense building blocks is a viable method for improving neural networks for computer vision.  
**Moving to sparser architectures is feasible and useful idea in general.**  

## Introduction
The biggest gains in object-detection come from the synergy of deep architecture and classical computer vision.  

## Related Work
### CNN
standard structure: stacked conv layers(each followed by norm layer and max-pooling)+fc layers.  
**Trend**: more,larger layers + dropout

### multiple scales
Inception model learns filters of different sizes to handle multiple scales.  
GoogLeNet = 22-layer repeated Inception layers

### Network-in-Network
**Network-in-Network approach**: additional 1x1 conv layers followed typically by ReLU**.  
* an approach to increase the representational power of neural networks.
* remove computational bottlenecks -> larger depth&width available

### R-CNN
Regions with Convolutional Neural Networks: The current leading approach for object detection.  
#### R-CNN: detection problem -> 2 subproblems
1. utilize low-level cues for potential object proposals.  
2. use CNN classifiers to identify object categories at these locations.  
