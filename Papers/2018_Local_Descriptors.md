
# Local Descriptors Optimized for Average Precision

[CVPR 2018 Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_Local_Descriptors_Optimized_CVPR_2018_paper.pdf)

## Summary
* optimizing descriptor matching -> improve the learning of local featrue descriptors.  
* optimize Average Precision(AP, ranking-based retrieval performance metric)
* a listwise learning to rank approach
* best patch verification, patch retrieval and image matching
* worked for binary 

## Introduction
### Why local feature descriptors?
to replace handcrafted feature engineering
### What's the problem?
handcrafting in learning objectives(for descriptors), hard to fit in large local feature mapping pipelines
### What we did?
improve the learning of local feature descriptors by optimizing better objectives
### How we did that?
design learning objectives in accordance with other pipeline components.  
### Why listwise learning to rank?
feature matching can be formulated as nearest neighbor retrieval
