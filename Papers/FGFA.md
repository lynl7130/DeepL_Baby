
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
* deep CNN: generate a set of feature maps over the whole input
* shallow detection-specific network: generate results from feature maps

### Why this won't work on videos?

