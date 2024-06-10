import environments_fully_observable 
import environments_partially_observable
import numpy as np
from  tqdm import trange
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from agent import *
tf.random.set_seed(0)
random.seed(0)
import math
np.random.seed(0)
import wandb
import random

# function to standardize getting an env for the whole notebook
def get_env(n=1000):
    # n is the number of boards that you want to simulate parallely
    # size is the size of each board, also considering the borders
    # mask for the partially observable, is the size of the local neighborhood
    size = 7
    e = environments_fully_observable.OriginalSnakeEnvironment(n, size)
    # or environments_partially_observable.OriginalSnakeEnvironment(n, size, 2)
    return e

def main():
    reward_history_conv=[0]
    EPOCHS=1500
    env_ = get_env()
    agent=DQN(env=env_)
    step=1
    for e in range(EPOCHS):
        if e%50==0:
            print(f"{e}/{EPOCHS} - {np.mean(reward_history_conv[-30:]) or 0}", end="\n")
        
        #if e%400==0 and e!=0:
        #    step=step+1        

        state=env_.to_state()
        state=tf.reshape(state, (1000, -1))

        value_q=agent.Q_net(state)
        #original_probs = agent.Q_net(state)
        #pred = original_probs / tf.reduce_sum(original_probs, axis=-1, keepdims=True)
        #samples = tf.random.categorical(tf.math.log(pred), 1, dtype=tf.int32)[:, 0]
        samples=tf.argmax(value_q, axis=1)
        actions=tf.reshape(samples, (samples.shape[0],1))

        masks=env_.check_actions(actions)

        rewards= env_.move(actions)
        new_state = tf.constant(env_.to_state())
        new_state=tf.reshape(new_state, (1000, -1))

        agent.learn(state, new_state, samples, rewards) # this optimie the policy function tarting from information sampled
        if e > 50: reward_history_conv.append(np.mean(rewards))
    plt.plot(reward_history_conv)

if __name__=='__main__':
    main()