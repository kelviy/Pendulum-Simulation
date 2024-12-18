"""
running openaigym model

args:
- "<num>" - first argument (for determining how many simulations to run)
- "sim"
- "model-train"
- "model-run"

"""
import sys
from openAIGym.Simulation_OOP import Simulation

def main():
    # extracts arguments
    args = []
    if len(sys.argv) == 1:
        print("No arguments passed: using default arguments")
        args = ["15", "sim"]
    else:
        args = sys.argv[1:]

    print(f"Arguments passed: { ' '.join( str(arg) for arg in args ) }")

    sim = Simulation()

    sim_no = int(args[0])

    if "sim" in args:
        sim.run_episodes(sim_no, render = True)

    if "model-train" in args:
        sim.train_model()
        sim.save_model() 
        
    if "model-run" in args:
        sim.load_model("../Output/simple_reinforce_model/001/SRM.keras")
        sim.run_episodes(sim_no, render = True)

if __name__ == "__main__":
    main()
