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
#### Categories:  
* static or movable.  
* interactable or non-interactable.  
  
#### Interactable objects:
* included in the metadata that server sends.
* 87 categories of interactable objects.
* each interactable object contains a number of variation.  
* a **randomizer** that can be used to change the location of the objects.  
* the variant of the object that is selected per scene is deterministic.  

### Object Visibility
**object visible** = in camera view + within a threshold of distance(default: 1 meter) from camera to the closest point of the object.  
* object visibility != visibility in the image.

## Architecture
[The overall architecture](https://arxiv.org/pdf/1712.05474.pdf#page=3)
### Two components:  
* A set of scenes built within the Unity Game engine.  
* a lightweight Python API that interacts with the game engine.  

### Procedure
Action executed in Unity: screen capture + Json metadata -> Python Flask service.  
Received -> an Event Object(a numpy array:screen capture + a dictionary: current states).  
Unity wait for a controller.step() from the Python service.  

## Usage
target-driven navigation using DeepRL(fine-tuning in real world)   
visual semantic planning using deep successor representations(generalized)  
a GAN for generating occluded regions of objects  
interactive visual question answering  
