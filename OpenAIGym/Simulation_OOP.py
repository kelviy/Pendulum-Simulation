import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])


    def run_episodes(self, num, render=False):
        for i in range(num):
            if render == True:
                self.ax.set_title(f"Simulation {i}")
                self.run_episode(i, render=True)
            else:
                self.run_episode(i)
        self.print_statistics()


    def run_episode(self, episode, render=False):
        episode_rewards = 0
        obs, info = self.env.reset(seed=episode)

        for step in range(200):  # Maximum number of steps in an episode
            action = self.basic_policy(obs)
            obs, reward, done, truncated, info = self.env.step(action)
            episode_rewards += reward

            # Render the environment and update the plot
            if render==True:
                self.img_plot.set_data(self.env.render())  # Update the image data
                self.fig.canvas.draw()  # Redraw the figure
                self.fig.canvas.flush_events()
                plt.pause(0.001)  # Pause to create the animation effect

            # if done or truncated:
            #     break

        self.totals.append(episode_rewards)

    def basic_policy(self, obs):
        angle = obs[2]
        return 0 if angle < 0 else 1

    def play_one_step(self, env, obs, model, loss_fn):
        with tf.GradientTape() as tape:
            left_proba = model(obs[np.newaxis])
            action = (tf.random.uniform([1, 1]) > left_proba)
            y_target = tf.reduce_mean(loss_fn(y_target, left_proba))
            loss = tf.reduce_mean(loss_fn(y_target, left_proba))

            grads = tape.gradient(loss, model.trainable_variables)
            obs,reward, done, truncated, info = env.step(int(action))
            return obs, reward, done, truncated, grads
        
    def play_multiple_episodes(self, env, n_episodes, n_max_steps, model, loss_fn):
        all_rewards = []
        all_grads = []

        for episode in range(n_episodes):
            current_rewards = []
            current_grads = []

            obs, info = env.reset()

            for step in range(n_max_steps):
                obs, reward, done, truncated, grads = self.play_one_step(env, obs, model, loss_fn)
                current_rewards.append(reward)
                current_grads.append(reward)
                current_grads.append(grads)

                if done or truncated:
                    break
            
            all_rewards.append(current_rewards)
            all_grads.append(current_grads)

            return all_rewards, all_grads
        
    def discount_rewards(self, rewards, discount_factor):
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * discount_factor
        return discounted

    def discount_and_normalize_rewards(self, all_rewards, discount_factor):
        all_discounted_rewards = [self.discount_rewards(rewards, discount_factor)
                                for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std
                for discounted_rewards in all_discounted_rewards]



    def print_statistics(self):
        # Calculate episode statistics
        mean_total = np.mean(self.totals)
        std_total = np.std(self.totals)
        min_total = min(self.totals)
        max_total = max(self.totals)

        print(f"Mean: {mean_total}, Std: {std_total}, Min: {min_total}, Max: {max_total}")