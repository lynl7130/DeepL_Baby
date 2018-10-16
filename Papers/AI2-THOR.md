# AI2-THOR

[paper](https://arxiv.org/pdf/1712.05474.pdf)

## Summary
enables research in:  
* deep reinforcement learning
* imitation learning
* learning by interaction
* planning
* visual question answering
* unsupervised representation learning
* object detection
* segmentation
* learning models of cognition
* ...

## Introduction
Key element: interact with the environment and learn from those interactions.  
learning from still images/videos -> human-like learning.  

### What's in AI2-THOR?
120 scenes spanning four different categories: kitchen, living room, bedroom and bathroom.  
  
### AI2-THOR's features:  
1. actionable objects.  
2. near photo-realistic.  
3. no bias(designed by 3D artists!)  
4. a Python API to interact with Unity3D engine: navigation, applying force, object interaction and physics modeling.  
  
### Why AI2-THOR?
a scalable, fast and cheap proxy for real world experiments in different types of scenarios.  

## Related Platforms
[a summary table](https://arxiv.org/pdf/1712.05474.pdf#page=3)
### Game environments: 
not photo-realistic / expose the full environment to the agent!  
* ATARI Learning Environment
* ViZDoom
* TorchCraft
* ELF
* DeepMind Lab
* OpenAI Universe

### non-photo-realistic simulated environments
* UETorch
* Project Malmo
* SceneNet

### for autonomous driving
* SYNTHIA
* Virtual KITTI
* TORCS
* CARLA

### other synthetic indoor environments
suitable for navigation due to lack of actionable objects and an interaction API.  
* HoME
* House3D
* MINOS
* SceneNet RGBD  
**Furthermore, AI2-ThOR is integrated with a physics engine!**

## Concepts
### Scene
a virtual room that an agent can navigate in and interact with.  

### Agent
* radius: 0.2m, height: 1.8m
* cannot pass through physical objects.

### Action
Actions fail if pre-conditions are not met.

### Object
Categories:  
* static or movable.  
* interactable or non-interactable.  





