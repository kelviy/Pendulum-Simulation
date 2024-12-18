import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import numpy as np
from pathlib import Path

class BasicModel:

    # hyper-parameters
    n_interations = 150
    n_episodes_per_update = 10
    n_max_steps = 200
    discount_factor = 0.95

    def policy(self, obs):
        angle = obs[2]
        return 0 if angle < 0 else 1

class BasicNN:
    """
    Simple Neural Network using the REINFORCE algorithm
    """

    # hyper-parameters
    n_iterations = 150
    n_episodes_per_update = 10
    n_max_steps = 200
    discount_factor = 0.95


    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.binary_crossentropy
        
    def policy(self, obs):
        left_proba = self.model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        return int(action)

    def training_loop(self, env):
        for iteration in range(self.n_iterations):
            all_rewards, all_grads = BasicNN.play_multiple_episodes(
                env, self.n_episodes_per_update, self.n_max_steps, self.model, self.loss_fn
                )
            
            all_final_rewards = BasicNN.discount_and_normalize_rewards(all_rewards, self.discount_factor)

            #debug
            total_rewards = sum(map(sum, all_rewards))
            print(f"\rIteration: {iteration + 1}/{self.n_iterations},"
                  f" mean rewards: {total_rewards / self.n_episodes_per_update:.1f}", end="")

            all_mean_grads = []
            for var_index in range(len(self.model.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [final_reward * all_grads[episode_index][step][var_index]
                    for episode_index, final_rewards in enumerate(all_final_rewards)
                        for step, final_reward in enumerate(final_rewards)], axis=0)
                all_mean_grads.append(mean_grads)

            self.optimizer.apply_gradients(zip(all_mean_grads, self.model.trainable_variables))

    # Variant of the REINFORCE algorithm implemented using Keras
    def play_one_step(env, obs, model, loss_fn):
        with tf.GradientTape() as tape:
            left_proba = model(obs[np.newaxis])
            action = (tf.random.uniform([1, 1]) > left_proba)
            y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
            loss = tf.reduce_mean(loss_fn(y_target, left_proba))

        grads = tape.gradient(loss, model.trainable_variables)
        obs, reward, done, truncated, info = env.step(int(action))
        return obs, reward, done, truncated, grads
        

    def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
        all_rewards = []
        all_grads = []
        for episode in range(n_episodes):
            current_rewards = []
            current_grads = []
            obs, info = env.reset()
            for step in range(n_max_steps):
                obs, reward, done, truncated, grads = BasicNN.play_one_step(
                    env, obs, model, loss_fn)
                current_rewards.append(reward)
                current_grads.append(grads)
                if done or truncated:
                    break

            all_rewards.append(current_rewards)
            all_grads.append(current_grads)

        return all_rewards, all_grads
            
    def discount_rewards(rewards, discount_factor):
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * discount_factor
        return discounted

    # subtract mean and divide by standard deviation
    def discount_and_normalize_rewards(all_rewards, discount_factor):
        all_discounted_rewards = [BasicNN.discount_rewards(rewards, discount_factor)
                                for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std
                for discounted_rewards in all_discounted_rewards]

    def save(self, version):
        model_name = "simple_reinforce_model"
        model_version = version 
        model_dir = Path() / ".." / "Output" / model_name / model_version  # Directory path
        model_path = model_dir / "SRM.keras"  # File path

        model_dir.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
        self.model.save(model_path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
