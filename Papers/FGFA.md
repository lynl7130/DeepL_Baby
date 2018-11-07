
# Flow-Guided Feature Aggregation for Video Object Detection

[ICCV 2017 paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Flow-Guided_Feature_Aggregation_ICCV_2017_paper.pdf)

## Summary
FGFA(Flow-Guided Feature Aggregation):  
an accurate and end-to-end learning framework for video object detection.  
* temporal coherence on feature level
* aggregate nearby features along the motion paths
* single-frame baselines in ImageNet VID

## Introduction

### State-of-art object detection method in still images
a two-stage structure:  
1. deep CNN: generate a set of feature maps over the whole input.  
2. shallow detection-specific network: generate results from feature maps.  

### Why this won't work on videos?
Deteriorated obejct apperances in videos! 
* motion blur
* video defocus
* rare poses
* ...

### box level methods
Box level methods: exploit temporal information in videos(multiple "snapshots").
1. apply object detectors in single frames.  
2. assemble the detected bounding boxes across temporal dimension in a dedicated post-processing step.  
* post-processing step relies on motion estimation(optical flow) and bounding box association rules(object tracking).  
* Improve performance by heuristic post-processing, not by principled learning!  
[Note for these methods](https://github.com/lynl7130/DeepL_Baby/tree/master/Papers/Box_Level_Methods)

### A principle way: end-to-end learning
Improve the detection or recognition quality by exploiting temporal information, in a principled way.  
1. feature matters.  
-> improve the per-frame feature learning by temporal aggregation.  
2. the features of the same object instance are usually not spatially aligned across frames due to video motion.  
-> model the motion during learning.   

### FGFA: end-to-end!
1. **feature extraction network**: individual frames -> per-frame feature maps.  
2. **optical flow network**: estimate motion between nearby frames and the reference frame.  
3. **feature warping**: warp nearby feature maps to the reference frame according to the flow motion.  
4. **feature aggregation**: adaptive weighting network: aggregate the warped feature maps.  
5. **detection network**: aggregated feature maps -> detection result on the reference frame.  
