from Simulation_OOP import Simulation

def main():
    sim = Simulation()
    sim.run_episodes(500, render = True)

if __name__ == "__main__":
    main()