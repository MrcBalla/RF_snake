import keras.api._v2.keras as K
import tensorflow as tf
import numpy as np
from environments_fully_observable import *
from environments_partially_observable import *

class Agent:
    def __init__(self, env: BaseEnvironment, batch_size=256, discount=0.99, clip_eps=0.1, step_size=1e-4,
                 actor_rep=15, critic_rep=1):
        self.discount = discount
        self.clip_eps = clip_eps
        self.actor_rep = actor_rep
        self.critic_rep = critic_rep
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Dense(5, activation=tf.nn.softmax,
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
        ])
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Dense(1, activation="linear")
        ])
        self.optimizer_actor = tf.optimizers.legacy.Adam(step_size)
        self.optimizer_critic = tf.optimizers.legacy.Adam(step_size)
        self.batch_size = batch_size



    def learn(self, states, new_states, samples, rewards, dones, masks):
        """
        Proximal Policy Optimization (PPO) implementation using TD(0)
        """
        rewards = np.reshape(rewards, (-1, 1))
        dones = np.reshape(dones, (-1, 1))
        actions = tf.one_hot(samples, depth=masks.shape[-1]).numpy()
        initial_probs = None
        val = self.critic(states)
        new_val = self.critic(new_states)
        reward_to_go = tf.stop_gradient(rewards + self.discount * new_val * (1-dones))
        td_error = (reward_to_go - val).numpy()

        for _ in range(self.actor_rep):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as a_tape:
                probs = self.actor(states[indexes])
                probs = probs * masks[indexes]
                probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                selected_actions_probs = tf.reduce_sum(probs * actions[indexes], axis=-1, keepdims=True)   # the more difficult part, look at gahter on tensorflow or similar stuff, it will do similarly
                if initial_probs is None: initial_probs = tf.convert_to_tensor(tf.stop_gradient(selected_actions_probs))
                importance_sampling_ratio = selected_actions_probs / initial_probs
                loss_actor = tf.minimum(
                    td_error[indexes] * importance_sampling_ratio,
                    td_error[indexes] * tf.clip_by_value(importance_sampling_ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                )
                loss_actor = tf.reduce_mean(-loss_actor)

            grad_actor = a_tape.gradient(loss_actor, self.actor.trainable_weights)
            self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor.trainable_weights))

        for _ in range(self.critic_rep):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as c_tape:
                val = self.critic(states[indexes])
                new_val = tf.stop_gradient(self.critic(new_states[indexes]))
                reward_to_go = tf.stop_gradient(rewards[indexes] + self.discount * new_val * (1-dones[indexes]))
                loss_critic = tf.losses.mean_squared_error(val, reward_to_go)[:, None]
                loss_critic = tf.reduce_mean(loss_critic)
            grad_critic = c_tape.gradient(loss_critic, self.critic.trainable_weights)
            self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic.trainable_weights))

