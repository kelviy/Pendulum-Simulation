import neat.parallel
from pendulum_engine import environment
import os
import neat
import multiprocessing
from pathlib import Path
import pickle

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        envi = environment.Environment()
        while envi.time <= 300:
            observations = envi.observe()
            output = net.activate(observations)
            score = envi.scoreCheck(output)
            genome.fitness += score

def eval_genome(genome, config):
    score = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
        
    envi = environment.Environment()
    while envi.time <= 300:
        observations = envi.observe()
        output = net.activate(observations)
        # print(output)
        score += envi.scoreCheck(output)
        # print(score, "at ", envi.time)

    # print(score)
    return score

def main(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    path = Path() / '..' / 'Output' / 'models'
    path.mkdir(parents=True, exist_ok=True)

    #p.add_reporter(neat.Checkpointer(2, filename_prefix= (path / 'neat-').absolute().as_posix()))
    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, 10)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    with open(path / 'neat-run', "wb") as f:
        pickle.dump(winner_net, f)
        f.close()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_pendulum')
    main(config_path) # <-- Trains model and runs after training
