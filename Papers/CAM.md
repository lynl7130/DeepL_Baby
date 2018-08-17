
![Project Link](http://cnnlocalization.csail.mit.edu/)

# Summary
Purpose CAM(Class Activation Mapping) for CNNs with global average pooling.  
CAM can:
* Let classification-trained CNNs to learn to perform object localization without annotation
* Let Network localize the discriminative image regions despite just being trained for solving classification task
* Generate to other tasks(generic localizable deep features)
	
### Why CAM?
Convolutional layers: could localize objects without supervision, but the ability is lost when fully-connected layers are used for classification!

Networks avoiding fully-connected layers:  
1. Network in Network(NIN) . 
2. GoogLeNet .  
Minimizing parm number & maintaining performance  

### What is Global average pooling(GAP)?
![](https://latex.codecogs.com/gif.latex?%28x%2C%20y%29): spatial location.     
![](https://latex.codecogs.com/gif.latex?f_k): activation of unit k of the last conv layer.  

For unit k, the result of performing global average pooling:

![](https://latex.codecogs.com/gif.latex?F%5E%7Bk%7D%20%3D%20%5Csum%20_x_y%20f_k%28x%2C%20y%29)
* function: a structural regularizer(tradition), but can do more after tweaking: 
1. The network could retain the localization ability until the final layer, even those not originally trained for!
2. Localizability could transfer to other dataset & tasks

# Class Activation Mapping
* Class Activation Map for a particular category indicates: the discriminative image regions used by CNN to identify that category
* Class Activation Mapping: project back the weights of the output layer on to the convolutional feautre maps
conv layers -> GAP -> fully-connected layer(softmax, regression, other losses...)

For a given class c, the input to the softmax:

![](https://latex.codecogs.com/gif.latex?S_c%20%3D%20%5Csum%20_k%20%5Comega%20_k%5EcF_k%20%3D%20%5Csum%20_%7Bx%2Cy%7D%20%5Csum%20_k%20%5Comega%20_k%5Ecf_k%20%28x%2C%20y%29)

Class Activaton Map: 

![](https://latex.codecogs.com/gif.latex?M_c%28x%2C%20y%29%20%3D%20%5Csum%20_k%20%5Comega%20_k%5Ecf_k%20%28x%2C%20y%29)

![](https://latex.codecogs.com/gif.latex?S_c%20%3D%20%5Csum%20_%7Bx%2C%20y%7D%20M_c%20%28x%2C%20y%29)

**Thus Class Activation Map directly indicates the importance of the activation at spatial grid leading to the classification of an image to class c.**
By **upsampling** the map to the size of input image, could identify the image regions most relevant to the particular category  

### Why GAP, not GMP(global max pooling)?
* GAP: encourages the network to identify the extent of the object, because the value can be maximized by finding all discriminative parts of an object
* GMP: encourages to identify just one discriminative part; low scores for all image regions except the most discriminative one do not impact the score as perform a max

# Related Work(under construction)
### Weakly-supervised object localization
### Visualizing CNNs
