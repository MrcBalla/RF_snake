import keras.api._v2.keras as K
import tensorflow as tf
import numpy as np
from environments_fully_observable import *
from environments_partially_observable import *
import environments_fully_observable 
import math
    
class PPO:
    def __init__(self, env:environments_fully_observable, step_size=0.001, batch_size=256) -> None:
        self.discount = 0.95
        self.clip_eps = 0
        self.actor_rep = 20
        self.critic_rep = 5
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

    def save(self,path):
        tf.keras.saving.save_model(self.actor, filepath=path+"PPO_actor.h5",save_format='h5')
        tf.keras.saving.save_model(self.critic, filepath=path+"PPO_critic.h5",save_format='h5')

    def learn(self, states, new_states, samples, rewards):
        rewards=np.reshape(rewards, (-1,1)) # reshape of rewards
        val = self.critic(states) # compute values for creitic, so value function approximation
        actions=tf.one_hot(samples, depth=5) # create one hot encoding vector for action
        initial_probs=None
        new_val = self.critic(new_states) # create value function at new state 
        reward_to_go = tf.stop_gradient(rewards + self.discount * new_val) # stop gradient 
        td_error = (reward_to_go - val).numpy() # td error phase
        
        for _ in range(self.actor_rep):
            indexes=np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as a_tape:
                probs = self.actor(tf.gather(states, indexes))
                probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                selected_actions_probs = tf.reduce_sum(probs * np.array(tf.gather(actions, indexes)), axis=-1, keepdims=True) 
                if initial_probs is None: initial_probs = tf.convert_to_tensor(tf.stop_gradient(selected_actions_probs))
                importance_sampling_ratio = selected_actions_probs / initial_probs
                loss_actor = tf.minimum(
                    td_error[indexes] * importance_sampling_ratio,
                    td_error[indexes] * tf.clip_by_value(importance_sampling_ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                )
                loss_actor = tf.reduce_mean(-loss_actor)
                #print("loss actor:{}".format(loss_actor))
            grad_actor = a_tape.gradient(loss_actor, self.actor.trainable_weights)
            self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor.trainable_weights))

        for _ in range(self.critic_rep):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as c_tape:
                val = self.critic(tf.gather(states, indexes))
                new_val = tf.stop_gradient(self.critic(tf.gather(new_states, indexes)))
                reward_to_go = tf.stop_gradient(rewards[indexes] + self.discount * new_val)
                loss_critic = tf.losses.mean_squared_error(val, reward_to_go)[:, None]
                loss_critic = tf.reduce_mean(loss_critic)
                #print("loss critic:{}".format(loss_critic))
            grad_critic = c_tape.gradient(loss_critic, self.critic.trainable_weights)
            self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic.trainable_weights))
        

class PPO_conv:
    def __init__(self, env:environments_fully_observable, step_size=0.001, batch_size=256) -> None:
        self.discount = 0.95
        self.clip_eps = 0
        self.actor_rep = 20
        self.critic_rep = 5
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Conv2D(16, kernel_size=(2,2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Conv2D(5, kernel_size=(3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(5, activation=tf.nn.softmax,
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
        ])
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Conv2D(16, kernel_size=(3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Conv2D(5, kernel_size=(5,5)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
        ])
        self.optimizer_actor = tf.optimizers.legacy.Adam(step_size)
        self.optimizer_critic = tf.optimizers.legacy.Adam(step_size)
        self.batch_size = batch_size

    def save(self,path):
        tf.keras.saving.save_model(self.actor, filepath=path+"PPO_actor_conv.h5",save_format='h5')
        tf.keras.saving.save_model(self.critic, filepath=path+"PPO_critic_conv.h5",save_format='h5')


    def learn(self, states, new_states, samples, rewards):
        rewards=np.reshape(rewards, (-1,1)) # reshape of rewards
        val = self.critic(states) # compute values for creitic, so value function approximation
        actions=tf.one_hot(samples, depth=5) # create one hot encoding vector for action
        initial_probs=None
        new_val = self.critic(new_states) # create value function at new state 
        reward_to_go = tf.stop_gradient(rewards + self.discount * new_val) # stop gradient 
        td_error = (reward_to_go - val).numpy() # td error phase
        
        for _ in range(self.actor_rep):
            indexes=np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as a_tape:
                probs = self.actor(tf.gather(states, indexes))
                probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
                selected_actions_probs = tf.reduce_sum(probs * np.array(tf.gather(actions, indexes)), axis=-1, keepdims=True) 
                if initial_probs is None: initial_probs = tf.convert_to_tensor(tf.stop_gradient(selected_actions_probs))
                importance_sampling_ratio = selected_actions_probs / initial_probs
                loss_actor = tf.minimum(
                    td_error[indexes] * importance_sampling_ratio,
                    td_error[indexes] * tf.clip_by_value(importance_sampling_ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                )
                loss_actor = tf.reduce_mean(-loss_actor)
                #print("loss actor:{}".format(loss_actor))
            grad_actor = a_tape.gradient(loss_actor, self.actor.trainable_weights)
            self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor.trainable_weights))

        for _ in range(self.critic_rep):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as c_tape:
                val = self.critic(tf.gather(states, indexes))
                new_val = tf.stop_gradient(self.critic(tf.gather(new_states, indexes)))
                reward_to_go = tf.stop_gradient(rewards[indexes] + self.discount * new_val)
                loss_critic = tf.losses.mean_squared_error(val, reward_to_go)[:, None]
                loss_critic = tf.reduce_mean(loss_critic)
                #print("loss critic:{}".format(loss_critic))
            grad_critic = c_tape.gradient(loss_critic, self.critic.trainable_weights)
            self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic.trainable_weights))
        
class DQN:
    def __init__(self, env:environments_fully_observable, step_size=0.001, batch_size=256) -> None:
        self.discount = 0.95
        self.clip_eps = 0
        self.Q_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Dense(5, activation='softmax',
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
        ])
    
        self.Q_net_optim = tf.optimizers.legacy.Adam(step_size)
        self.batch_size = batch_size
    
    def save(self,path):
        tf.keras.saving.save_model(self.Q_net, filepath=path+"Qnet.h5",save_format='h5')


    def learn(self, states, new_states, actions, rewards):
        rewards=np.reshape(rewards, (-1,1)) # reshape of rewards
        actions=tf.one_hot(actions, depth=5, dtype=tf.int32) 
        actions=tf.cast(actions, tf.bool)

        initial_probs=None

        for _ in range(20):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as c_tape:
                actions_index=tf.gather(actions, indexes)

                q_v=tf.stop_gradient(self.Q_net(tf.gather(new_states, indexes)))
                y=tf.stop_gradient(rewards[indexes]+self.discount*tf.reshape(tf.reduce_max(q_v, axis=1), (256,1)))

                y_hat=(tf.reshape(tf.boolean_mask(self.Q_net(tf.gather(states, indexes)), actions_index), (256,1)))
                
                loss = tf.losses.mean_squared_error(y, y_hat)[:, None]
                loss = tf.reduce_mean(loss)
        
            
            grad_q_net = c_tape.gradient(loss, self.Q_net.trainable_weights)
            self.Q_net_optim.apply_gradients(zip(grad_q_net, self.Q_net.trainable_weights))

class DQN_conv:
    def __init__(self, env:environments_fully_observable, step_size=0.001, batch_size=256) -> None:
        self.discount = 0.95
        self.clip_eps = 0
        self.Q_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Conv2D(16, kernel_size=(3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Conv2D(5, kernel_size=(5,5)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))
        ])
    
        self.Q_net_optim = tf.optimizers.legacy.Adam(step_size)
        self.batch_size = batch_size
    
    def save(self,path):
        tf.keras.saving.save_model(self.Q_net, filepath=path+"Qnet_conv.h5",save_format='h5')

    def learn(self, states, new_states, actions, rewards):
        rewards=np.reshape(rewards, (-1,1)) # reshape of rewards
        actions=tf.one_hot(actions, depth=5, dtype=tf.int32) 
        actions=tf.cast(actions, tf.bool)

        initial_probs=None

        for _ in range(10):
            indexes = np.random.choice(range(0, len(states)), min(self.batch_size, len(states)), replace=False)
            with tf.GradientTape() as c_tape:
                actions_index=tf.gather(actions, indexes)

                q_v=tf.stop_gradient(self.Q_net(tf.gather(new_states, indexes)))
                y=tf.stop_gradient(rewards[indexes]+self.discount*(tf.reduce_max(q_v, axis=1)))

                y_hat=(tf.reshape(tf.boolean_mask(self.Q_net(tf.gather(states, indexes)), actions_index), (256,1)))
                
                loss = tf.losses.mean_squared_error(y, y_hat)[:, None]
                loss = tf.reduce_mean(loss)
        
            
            grad_q_net = c_tape.gradient(loss, self.Q_net.trainable_weights)
            self.Q_net_optim.apply_gradients(zip(grad_q_net, self.Q_net.trainable_weights))
        

def compute_neighbours(pos):
    pos_return_0 = np.copy(pos)
    pos_return_1 = np.copy(pos)
    pos_return_2 = np.copy(pos)
    pos_return_3 = np.copy(pos)

    pos_return_0[0] = pos_return_0[0]+1 # move at right
    pos_return_1[1] = pos_return_1[1]+1 # move up
    pos_return_2[0] = pos_return_2[0]-1 # move at left
    pos_return_3[1] = pos_return_3[1]-1 # move down

    return [pos_return_0, pos_return_1, pos_return_2, pos_return_3]

def BFS_search(state):
    visited={}
    selected_path={}
    heads = np.argwhere(state[:,:,1] == 1)[0] # position of the head
    fruit = np.argwhere(state[:,:,3] == 1)[0] # position of the fruit
    boundary = np.argwhere(state[:,:,0] == 0)
    q=[heads]

    for a in range(8):
        for b in range(8):
            visited[(a,b)]=False

    visited[tuple(heads)]=True

    selected_path[tuple(heads)]=None


    while len(q)!=0:
        node = q.pop(0)
        new_pos=compute_neighbours(node)

        for pos in new_pos:
            if tuple(pos)==tuple(fruit):
                selected_path[tuple(pos)]=tuple(node)
                visited[tuple(pos)]
                node=[]
                return pos, selected_path
            if not (np.any(np.all(boundary == pos, axis=1))) and visited[tuple(pos)]==False:
                selected_path[tuple(pos)]=tuple(node)
                visited[tuple(pos)]=True
                q.append(pos)