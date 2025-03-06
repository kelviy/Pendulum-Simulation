"""
Purpose is to Test:
- NEAT model 
- Pendulum Simulation

Options/Arguments for testing script are listed as follows: 

"sim" - for running pendulum simulation

"model-train" - for training model
"model-run" - for running model on engine

"stat" - for enabling output of simulation statistics 

default  args are: "engine" "stat"
"""

import sys
from pendulum_engine import environment
import os
import neat_model.NEAT as NEAT
from pathlib import Path
import pickle


def main():
    # extracts arguments
    args = []
    if len(sys.argv) == 1:
        print("No arguments passed: using default arguments")
        args = ["engine", "stat"]
    else:
        args = sys.argv[1:]

    print(f"Arguments passed: { ' '.join( str(arg) for arg in args ) }")
    
    # runs arguments
    env = environment.Environment()

    if "sim" in args:
        env.run()

    if "stat" in args:
        env.plot()

    if "model-train" in args:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, Path() / 'config_pendulum')
        NEAT.main(config_path)
    
    if "model-run" in args:
         # Unpickle saved winner
        genome_path = Path() / '..' / 'Output' / 'models'/ 'neat-run'
        with open(genome_path, "rb") as f:
           winner = pickle.load(f)

        # Convert loaded genome into required data structure
        env.run_model(winner)


if __name__ == "__main__":
    main()

