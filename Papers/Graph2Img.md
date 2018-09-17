
# Summary
An end-to-end method for generating imgs from scene graphs.  
* explicitly reasoning about objects and their relationships.  
* method:  
1. uses graph convolution to process input graphs.  
2. layout <- predicting bounding boxes and segmentation masks for objects.  
3. layout -> an image with a cascaded refinement network.  
  
* train the network!  
trained adversarially against a pair of discriminators to ensure realistic outputs.  
* validation
on Visual Genome and COCO-Stuff(qualitative results, ablations and user studies).  
