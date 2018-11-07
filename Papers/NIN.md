
# Network In Network

[ICLR 2014 paper](https://arxiv.org/pdf/1312.4400.pdf)

## Summary
NIN: a deep network **structure**.
### conventional conv layers?
linear filters followed by nonlinear activation function to scan the input.  
### In NIN?
micro NNs with more complex structures to abstract the data within the receptive field.  
### Micro NNs?
a multilayer perceptron(a function approximator) -> **enhance local modeling**
### How to use NIN?
1. slide the micro NNs over the input like in CNN -> feature maps.  
2. feature maps -> next layer
### Replace FC with GAP?
Global Average Pooling is easier to interpret and less prone to overfitting.  
**GAP could be used due to better local modeling!**
