
# Deep Learning Software


## CPU vs. GPU
**CPU**:  
* fewer cores, but each core is much faster and much more capable.   
* great at sequential tasks.  
* cache is relative small, using system RAMs(12/16/32G).  
**GPU**:  
* more cores, but each core is much slower and "dumber".  
* great for parallel tasks.  
* have own cache and own RAM built in the chip, given it's slow to comunicate between system RAM and GPU.   

### NVDIA vs. AMD
NVDIA wins!

### Example: Matrix Multiplication -> something suitable for GPUs!
```
    B            C             C
A       x    B        =    A
```
imagine having AxC cores doing the same thing:  
read in a row and a column then output a number!  => super fast with GPU.  

### Programming GPUs
#### CUDA(NVDIA only)
* Write C-like code that runs directly on the GPU  
complicated! => use APIs:  
* Higher-level APIs: cuBLAS, cuFFT, cuDNN, etc.  
#### OpenCL
* similar to CUDA, but runs on anything: NVDIA GPUs, AMDs, CPUs  
* Usually slower: has not yet been optimized  
  
#### Learn to write!
* Udacity: Intro to Parallel Programming  

## Deep Learning Frameworks
### Caffe/Caffe2
### Theano/Tensorflow
### Torch/PyTorch
