
# Image Classification

### Semantic Gap
**semantic**: car is the semantic label assigned to an image.  
**gap**: semantic idea of an image <-> the numbers the computer sees.  

### Challenges
  1. Viewpoint Variation.  
  2. Illumination.  
  3. Deformation.  
  4. Occlusion.  
  ...  

### Solve: Data-Driven Approach

Collect Data -> Get Classifier -> Evaluate Classifier.  

Image Classfication APIs:
```
Model train(input_images, labels)
Labels predict(model, test_images)
```

### First classifier: Nearest Neighbor
**train**: Memorize all data and labels.
**predict**: predict the label of the most similar training image. 
Example Dataset: CIFAR10  
* What does similar mean?  
   **Distance Metric** to compare images:  
   1. L1 distance(Manhattan distance):  
   ![](https://latex.codecogs.com/gif.latex?d_1%28I_1%2C%20I_2%29%20%3D%20%5Csum_p%20%7C%7BI_1%5Ep-I_2%5Ep%7D%7C)  
   2. L2 distance(Euclidean distance):  
   ![](https://latex.codecogs.com/gif.latex?d_2%28I_1%2C%20I_2%29%20%3D%20%5Csqrt%7B%5Csum_p%20%28%7BI_1%5Ep-I_2%5Ep%7D%29%5E2%7D)  
* How fast are training and prediction?  
  Train: O(1), predict: O(N).  
  **Bad!** Want classifiers that are fast at prediction; slow for training is ok.  
* What does this look like?  
  split into colored regions!  
  Problem: noise points/ overfitting => **K-Nearest Neighbors**  

#### K-Nearest Neighbors
Instead of copying label from nearest neighbor, take majority vote from K closest points
K up => removing noise & boundary becoming smoothy