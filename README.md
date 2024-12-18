# Overview
This repository consists of 3 modules. 
- Pendulum Engine
- Neat Model
- OpenAIGym Implementation

![pendulum gif](resources/pendulum.gif "pendulum gif")
![pendulum simulation](resources/pendulum.png "pendulum simulation")

![simulation statistics](resources/game_statistics.png "simulation statistics")

### Pendulum Engine
Basic pendulum physics simulator that is based on Euler's Method to approximate a single pendulum's motion with respect to a horizontal force acting on the base of the pendulum. 
The simulation is implemented using Pygame and Numpy. This pendulum engine acts as an environment for reinforcement learnings models to train and run on. Disclaimer, the implementation's very rudimentary in it's present state. The engine has a rough implmentation of a statistics reporter and a scoring certeria (points awarded based on pendulum being balanced upwright)

### Neat Model 
Using the neat-python library, a NEAT model is able to run and train on the Pendulum Engine. However, the model performs undesired actions when trained. Most noticably spinning the pendulum as fast as it can to earn points. 

### OpenAIGym Implementation
In its current form it does not integrate with the Pendulum Engine. There are future plans to implement this. Currently it is a Neural Network that uses the REINFORCE algorithm implemented using Keras.
The environmemt the model interacts with is the OpenAIGym environment. 

# Running and Testing
The three components in this repository is split into modules and can be accessed as such. There are two scripts that can run and be used to test the models. However make sure to install relevent dependencies before running - which can be found in the pyproject.toml.  

### Running NEAT
The following commands can be used to run the pendulum engine and/or NEAT Model depending on arguments passed. 
1. cd into src directory
2. run `python3 -m neat_model.neat-tests` which would run the pendulum simulation with statistics output as default. Arguments can be passed to change the behaviour

Arguments are as follows:
"engine" - for running pendulum simulation

"model-train" - for training model
"model-run" - for running model on engine

"stat" - for enabling output of simulation statistics 

default  args are: "engine" "stat"

### Running OpenAIGym
The following commands can be used to run the OpenAIGym implementation
1. cd into src directory
2. run `python3 -m openAIGym.gym-tests` which would run the simulations using a basic hard coded policy. Arguments can be passed to change the behaviour

Arguments are as follows:
- "<num>" - first argument (for determining how many simulations to run)
- "sim"
- "model-train"
- "model-run"

