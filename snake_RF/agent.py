import keras.api._v2.keras as K
import tensorflow as tf
import numpy as np
from environments_fully_observable import *
from environments_partially_observable import *
import environments_fully_observable 

class agent:
    def __init__(self, env:environments_fully_observable, step_size=0.001, batch_size=256) -> None:
        self.discount = 0.99
        self.clip_eps = 0
        self.actor_rep = 0
        self.critic_rep = 0
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

    
class new_agent:
    def __init__(self, env:environments_fully_observable, step_size=0.001, batch_size=256) -> None:
        self.discount = 0.99
        self.clip_eps = 0
        self.actor_rep = 100
        self.critic_rep = 100
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(10, (2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Conv2D(15, (2,2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(5, activation=tf.nn.softmax,
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
        ])
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Conv2D(10, (2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Conv2D(15, (2,2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, 
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
        ])
        self.optimizer_actor = tf.optimizers.legacy.Adam(step_size)
        self.optimizer_critic = tf.optimizers.legacy.Adam(step_size)
        self.batch_size = batch_size

    def learn(self, states, new_states, samples, rewards):
        rewards=np.reshape(rewards, (-1,1))
        val = self.critic(states)
        new_val = self.critic(new_states)
        reward_to_go = tf.stop_gradient(rewards + self.discount * new_val)
        td_error = (reward_to_go - val).numpy()

        for _ in range(self.actor_rep):
            indexes=np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as a_tape:
                probs = self.actor(states[indexes])
                probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                selected_actions_probs = tf.reduce_sum(probs * np.array(tf.gather(samples, indexes)), axis=-1, keepdims=True) 
                initial_probs = tf.convert_to_tensor(tf.stop_gradient(selected_actions_probs))
                importance_sampling_ratio = selected_actions_probs / initial_probs
                loss_actor = tf.minimum(
                    td_error[indexes] * importance_sampling_ratio,
                    td_error[indexes] * tf.clip_by_value(importance_sampling_ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                )
                loss_actor = tf.reduce_mean(-loss_actor)
                print("loss actor:{}".format(loss_actor))
            grad_actor = a_tape.gradient(loss_actor, self.actor.trainable_weights)
            self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor.trainable_weights))

        for _ in range(self.critic_rep):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as c_tape:
                val = self.critic(states[indexes])
                new_val = tf.stop_gradient(self.critic(new_states[indexes]))
                reward_to_go = tf.stop_gradient(rewards[indexes] + self.discount * new_val)
                loss_critic = tf.losses.mean_squared_error(val, reward_to_go)[:, None]
                loss_critic = tf.reduce_mean(loss_critic)
                print("loss critic:{}".format(loss_critic))
            grad_critic = c_tape.gradient(loss_critic, self.critic.trainable_weights)
            self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic.trainable_weights))

    