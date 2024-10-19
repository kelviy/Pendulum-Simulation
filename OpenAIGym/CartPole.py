from Simulation_OOP import Simulation

def main():
    sim = Simulation()
    # sim.run_episodes(500, render = True)

    # sim.training_loop()
    # sim.save()

    sim.load_model("Output/simple_reinforce_model/0001/SRM.keras")
    sim.run_episodes(10, render = True)

if __name__ == "__main__":
    main()