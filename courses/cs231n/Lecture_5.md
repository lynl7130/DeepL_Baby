
# Convolutional Neural Networks
  

## Convolution Layer: keep the structure of image!
32x32x3 image -> 5x5x3 filter -> **Convolve** the filter with the image -> 28x28x1      activation map: 28x28xn
                                                  filter .              -> 28x28x1  
                                                  filter .              -> 28x28x1   
                                                  ...   
**Convolve**: slide over the image spatially, computing dot products.  
* Why Convolutional? Because it is related to convolution of two signals.  
  
### ConvNet
ConvNet: a sequence of Convolution Layers, interspersed with activation functions.
Image -> Low-level features -> Mid-level features -> High-level features -> Linearly separable classifier ->   

### Visualize filter: What is a filter looking for?   
high value(white value): where there are targets.  
  
### Details  
1. stride: skip how many pixels when moving once.  
2. Output size: (N-F)/stride + 1  
3. Common to zero pad the border: to get the desirable output size.  
4. Tricky paramter#! each filter has 5x5x3 + 1(bias) = 76 params => 76x10=760 params.   
* 1x1 convolution(F=1) layers make perfect sense.  
Common settings:  
F=2, S=2.  
F=3, S=2.  

### Pooling Layer
- makes the representation smaller and more manageable.  
- operates over each activation map independently.  
**Pooling does nothing anything with depth!!!**  
1. Max Pooling(find where the spike is)   
  max pool with 2x2 filters and stride 2: 4x4 -> 2x2  
2....

### Fully Connected Layer
32x32x3 image -> stretch to 3072x1  -> times 10x3072 weights -> 10x1 -> activation.  
Contains neurons that connect to the entire input volume, as in ordinary Neural Networks.  

## Summary
- Trend towards smaller filters and deeper architectures.  
- Trend towards getting rid of POOL/FC layers(just CONV).  
- Typical architectures look like:  
{(Conv - ReLU)*N - Pool?}*M - (FC-RELU)*K, SOFTMAX  
where N is usually up to 5, M is large , 0<=K<=2.  
* But recent advances chanllenge this paradigm!  
