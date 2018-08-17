
##[Project](https://embodiedqa.org/)

# Summary
### Problem definition:   
an agent is spawned at a random location and asked a question.  
Solve: navigate to explore env, gather inf, answer.   
### EQA dataset of visual questions and answers:  
    1. control the distribution of question-types and answers in dataset.  
    2. deter algorithms from exploiting dataset bias.  
    3. provide fine-grained breakdown of performance by skill.  
### A Hierarchical Model, navigator: navigation-> planner + controller.  
  Planner: selects actions or a direction.  
  Controller: selects a velocity and execute the primitive actions a variable of times.  
  * planner->controller->planner->controller->… until it decides to give answer.  
### How model comes:
    1. initialize agent with imitation learning .   
    2. fine-tune agent with reinforcement learning(for the express purpose of correct Q&A) .   
    3. evaluation:(also on unseen environments!)   
  * Designed evaluation Protocol for EmbodiedQA , evaluated agent in House3D 
### Collected human demonstrations(human controlling agent) as benchmark

# Problem definition:
Agent spawned randomly in an env -> asked a question -> perceives env(RGB image) and perform atomic actions->navigate and collect enough inf to answer question.  
### Sub-tasks:  
	1. Active perception(must learn to map visual_input-> action)
	2. Common Sense Reasoning(where the car will be?)
	3. Language Grounding(goal-driven view of grounding)
	4. Credit Assignment(all up to agent! It has to learn all by itself)
* training time paradigm: the training environments are assumed to be sufficiently instrumented .   
Test time: agents operate entirely from egocentric RGB vision!

# EQA dataset
Build EQA dataset on a pruned subset of House3D: layout reasonable, typical, medium size and have kitchen, living room, dining room, bedroom.  
### Question-Answer Generation
* Exclude objects and rooms from SUNCG that are obscure(e.g. loggia rooms) or difficult to see(e.g. too small). Merged some semantically similar object categories to reduce ambiguity
* Question-> functional program
	Location template(question type): "What room is the <OBJ> located in?"
	Sequence of elementary operations for location:
	Select(objects)->unique(objects)->query(location)
	i.e. get all objects from env -> retain objects that have a single instance in house -> generate a question by filling in the appropriate template
* EQA is easily extensible. EQA v1 consist of location, color, color_room, preposition(they have one <OBJ>, makes imitation learning available)
* Dataset bias:
	1. exclude questions for which the normalized entropy of answer distribution <0.5(e.g. all refrigerators are located in kitchens, so all answers are kitchen)
	2. exclude questions occurring fewer than four environments
	3. exclude questions hard for human: threshold based on experience
* Dataset-> train, val, test(no overlap in envs across splits)

# A Hierarchical Model for EmbodiedQA
### Overview of the Agent 
	• Four modules: vision, language, navigation, answering .   
	• Adaptive Computation Time(ACT) RNNs: allow RNNs to learn how many computational steps to take between receiving an input and emitting an output by back-propagating through a 'halting' layer.   
* Used ACT in navigation module to cleanly separate the decision between direction and velocity

### Vision
Input: 224x224 RGB images from House3D .  
Encoder network(CNN): 5x5 Conv, ReLU, BatchNorm, 2x2 MaxPool-> fixed size representation.  
Pretrain CNN under a multi-task pixel-to-pixel prediction framework, multiple network heads(outputs) to decode:  
1. original RGB values 2. semantic class 3. depth of each pixel 

### Language(Question) 
Encoder:2-Layer LSTMs with 128-dim hidden states.  
* separate encoders for navigation and answering modules!
E.g. What color is the chair in the kitchen?

### Navigation
ACT navigator: separate intention from actions, strengthening long-term gradient flows(planner have variable time steps between decisions) . 
Planner:  a,h <- PLNR(h, I, Q, a)   
	• a: action . 
	• h: hidden state, updated only at planner timesteps, encoded intention . 
	• I: encoding of the imaged observed . 
	• Q: question encoding . 
Controller: {0, 1} <- CTRL(h, a, I), if 1(visual input aligns with intent of planner) then CTRL is applied to the next frame; elif 0 or reached 5 times than return to Planner . 

### Question Answering
After the agent decides to stop or a maximum number of actions taken . 
The last 5 frames observed, LSTM encoding of the question -> softmax over 172 answers . 

### Imitation Learning and Reward Shaping
Two stage training process:
	1. Navigation and answering modules are independently trained using imitation/supervised learning on automatically generated expert demonstrations of navigation .  
	2. The entire architecture is jointly fine-tuned using policy gradients .   

* Independent pretraining via Imitation Learning
1.  Navigation module training: 
	 history encoding, question encoding, current frame -> the action that would keep it on the shortest path(given)
	- cross-entropy loss and train
	- 15 epochs, NO.1: backtrack 10 steps, add 10steps each epoch 
	- batch size: 5 ~ 20 questions, depending on path length
2.  Question answering module training:
	 question, frame seen on the shortest path -> correct answer 
	- standard cross-entropy 
	- 50 epochs, 
	- batch size: 20
* Target-aware Navigational Fine-tuning . 
	Why fine-tune? Navigation and answering modules are poor in dealing with each other . 
	Reward signals to navigator:
	1. question answering accuracy at the end of the navigation: 5 if correct/0 otherwise
	2. reward shaping term when getting closer to the target: 0.005*change of dist
* Training details
