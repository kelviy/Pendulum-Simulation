import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import openAIGym.model as model

class Simulation:

    def __init__(self):
        # env variables
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.obs, self.info = self.env.reset(seed=42)
        self.totals = []

        # plotting variables
        self.fig, self.ax = plt.subplots()
        self.img_plot = self.ax.imshow(self.env.render()) 

        # model
        self.model = model.BasicNN() 
        #self.model = model.BasicModel()


    # rendering / simulation purposes (multi simulations)
    def run_episodes(self, num, render=False):
        for i in range(num):
            if render == True:
                self.ax.set_title(f"Simulation {i}")
                self.run_episode(i, render=True)
            else:
                self.run_episode(i)
        self.print_statistics()


    # rendering / simulation purposes (single simulation)
    def run_episode(self, episode, render=False):
        episode_rewards = 0
        obs, info = self.env.reset(seed=episode)

        for step in range(200):  # Maximum number of steps in an episode
            action = self.model.policy(obs)
            obs, reward, done, truncated, info = self.env.step(int(action))
            episode_rewards += reward

            # Render the environment and update the plot
            if render==True:
                self.img_plot.set_data(self.env.render())  # Update the image data
                self.fig.canvas.draw()  # Redraw the figure
                self.fig.canvas.flush_events()
                plt.pause(0.001)  # Pause to create the animation effect

            if done or truncated:
                break

        self.totals.append(episode_rewards)

    def train_model(self):
        self.model.training_loop(self.env)

    def save_model(self):
        self.model.save("001")

    def load_model(self, path):
        self.model.load(path)

    def print_statistics(self):
        # Calculate episode statistics
        mean_total = np.mean(self.totals)
        std_total = np.std(self.totals)
        min_total = min(self.totals)
        max_total = max(self.totals)

        print(f"Mean: {mean_total}, Std: {std_total}, Min: {min_total}, Max: {max_total}")
