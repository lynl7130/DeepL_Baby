
## [Project](https://github.com/KaimingHe/deep-residual-networks)

# Summary

## deep residual learning framework:
    1. easier to optimize & gain accuracy with increased depth 
    2. lower complexity & better performance

# Why ResNet?

### Is learning better networks as easy as stacking more layers? NO!
1. vanishing/exploding gradients
- addressed:  
    normalized initialization & intermediate normalized layers for SGD with BP
2. **degradation**: with network depth increasing, accuracy gets saturated and then degrades rapidly
- not caused by overfitting!
- degration shows that: **the solvers might have difficulties in approximating identity mappings by multiple nonlinear layers**  
-> if identity mappings are optimal, the solvers might simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings  
-> if the optimal function is closer to an identity mapping than to a zero mapping, it might be easier for the solver to learn the function

### shortcut connections
* shortcut connections: skipping one or more layers  
* In ResNet: perform identity mapping, and outputs are added to the outputs of the stacked layers

# Deep Residual Learning

### Residual Learning
expect the layers to approximate a **residual function**:

![](https://latex.codecogs.com/gif.latex?F%28x%29%20%3A%3D%20H%28x%29%20-%20x)

### Identity Mapping by Shortcuts
adopt residual learning to every few stacked layers.  

a **building block** defined as:(element-wise addition)  
![](https://latex.codecogs.com/gif.latex?y%20%3D%20F%28x%2C%20%5C%7BW_i%5C%7D%29%20&plus;%20x)  
**if need dimensions matching:**  
![](https://latex.codecogs.com/gif.latex?y%20%3D%20F%28x%2C%20%5C%7BW_i%5C%7D%29%20&plus;%20W_sx)  
  
![](https://latex.codecogs.com/gif.latex?x): input vectors of the layer  
![](https://latex.codecogs.com/gif.latex?y): output vectors of the layer    
![](https://latex.codecogs.com/gif.latex?F%28x%2C%20%5C%7BW_i%5C%7D%29): the residual mapping to be learned    
  
<img src=https://github.com/lynl7130/Out-of-Context/blob/master/Notes/Building_block.png width=500 height=250 >

**For convolutional layers:**  
The element-wise addition is performed on two feature maps, channel by channel  

### Network Architectures

<img src=https://github.com/lynl7130/Out-of-Context/blob/master/Notes/ResNet_arc.png width=320, height=1121>

<img src=https://github.com/lynl7130/Out-of-Context/blob/master/Notes/ResNet_template.png>

# Related Work(under construction)
### Residual Representation
### Shortcut Connections
