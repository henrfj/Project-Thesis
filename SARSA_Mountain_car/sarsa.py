import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from coarse_coding import Coarse
from mountain_car import MountainCar

rng = np.random.default_rng()

class SARSA:
    def __init__(self, game=MountainCar(), coarse_coder=Coarse(), discount_factor=0.99, eligibility_decay_rate=0.95, num_actions=3, learning_rate=1e-3, time_penalty=0.01):
        self.base_game = game
        self.model = keras.models.Sequential([
            # the task indicates that only an output layer may be sufficient
            # keras.layers.Dense(64),
            # keras.layers.Dense(64),
            # keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_actions),
        ])
        self.num_actions = num_actions
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.MSE)
        self.discount_factor = discount_factor
        self.eligibility_decay_rate = eligibility_decay_rate
        self.time_penalty = time_penalty
        self.coarse_coder = coarse_coder
        self.prev_state = None
        self.prev_action = None
        self.eligibility_traces = None
    
    def game_over(self, s: np.ndarray, a: int, reward: float):
        '''
        Applies learning for the final step of the game.
        :param s: vector representing previous state
        :param a: chosen action in previous state
        :param reward: reward for final state
        '''
        target = np.zeros(self.num_actions)
        target[a] = reward * self.discount_factor
        target = tf.convert_to_tensor(target)
        mask = np.zeros(self.num_actions)
        mask[a] = 1
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        with tf.GradientTape() as tape:
            yhat = self.model(s.reshape(1, -1)) * mask # multiply by zero in positions other than the taken action
        prediction_gradient = tape.gradient(yhat, self.model.trainable_weights)
        if self.eligibility_traces is None:
            self.eligibility_traces = prediction_gradient
        else:
            self.eligibility_traces = [self.eligibility_decay_rate * trace + grad
                                        for trace, grad in zip(self.eligibility_traces, prediction_gradient)]
        yhat = tf.math.reduce_sum(yhat)
        gradient = [grad * (yhat - reward * self.discount_factor) for grad in self.eligibility_traces]
        self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_weights))


    def learn(self, s: np.ndarray, a: int, s_prime: np.ndarray, a_prime: int):
        '''
        Makes a single increment in the value function, given a state, an action, the next state and the next chosen action.
        :param s: vector representing state
        :param a: integer representing action
        :param s_prime: vector representing next state
        :param a_prime: int representing next action
        '''
        mask = np.zeros(self.num_actions)
        mask[a] = 1
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask2 = np.zeros(self.num_actions)
        mask2[a_prime] = 1
        mask2 = tf.convert_to_tensor(mask2, dtype=tf.float32)
        yhat_prime = self.model.predict(s_prime.reshape(1, -1))
        target = (self.discount_factor * yhat_prime - self.time_penalty) * mask2
        # sum to get rid of the (position-wise) connection between a and a_prime
        target = np.sum(target, axis=1)
        with tf.GradientTape() as tape:
            yhat = tf.math.reduce_sum(self.model(s.reshape(1, -1)) * mask, axis=1)
        prediction_gradient = tape.gradient(yhat, self.model.trainable_weights)
        if self.eligibility_traces is None:
            self.eligibility_traces = prediction_gradient
        else:
            self.eligibility_traces = [self.eligibility_decay_rate * trace + grad
                                        for trace, grad in zip(self.eligibility_traces, prediction_gradient)]
        gradient = [grad * (yhat - target) for grad in self.eligibility_traces]
        self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_weights))
    
    def get_action(self, s: np.ndarray, epsilon=0.2) -> int:
        '''
        Uses an epsilon-greedy strategy to select the next action, given a state
        :param s: vector representing current state
        :param epsilon: float between 0 and 1, representing the probability of choosing a random
        :returns: integer representing the chosen action
        '''
        if rng.random() < epsilon:
            return rng.choice(self.num_actions)
        values = self.model(s.reshape(1, -1))
        return np.argmax(values)
    
    def run_episode(self, epsilon=0.):
        '''
        Runs a single episode of the provided game, using the 
        :returns: The game in its final state
        '''
        game = deepcopy(self.base_game)
        game.__init__() # each episode should be randomized
        while not game.is_state_final():
            # print(game.x, game.car_vel)
            state = self.coarse_coder.get_coarse_code(
                game.car_pos,
                game.car_vel,
            ) # TODO: adjust to reflect game class
            action = self.get_action(state) # TODO: set epsilon
            if self.prev_state is not None and self.prev_action is not None:
                # no reward to take into account before the end of the episode
                self.learn(self.prev_state, self.prev_action, state, action)
            game.make_action(action - 1) # action should be from -1 to 1 rather than 0 to 2
            self.prev_state = state
            self.prev_action = action
        # Applying final reward
        score = 5 * game.car_pos # TODO: include this in the game class
        self.game_over(self.prev_state, self.prev_action, score)
        self.eligibility_traces = None
        return game
    
    def run_episodes(self, iterations=100, epsilon=0., epsilon_decay_rate=1., *args, **kwargs):
        games = []
        for _ in tqdm(range(iterations)):
            games.append(self.run_episode(epsilon, *args, **kwargs))
            epsilon *= epsilon_decay_rate
        return games


if __name__ == '__main__':
    agent = SARSA(
        coarse_coder=Coarse(tilings=4, tiling_dims=5),
        discount_factor=0.99,
        eligibility_decay_rate=0.9,
        num_actions=3,
        learning_rate=1e-3,
        )
    history = agent.run_episodes(100, epsilon=1, epsilon_decay_rate=0.97)

    input('Press Enter to show a game')

    def get_move(game):
        cc = agent.coarse_coder.get_coarse_code(game.car_pos, game.car_vel)
        action = agent.get_action(cc, epsilon=0.) - 1
        print(action)
        return action
    try:
        MountainCar().show_track(get_move, filename='run.mp4')
    except Exception as e: # currently the episode will run until the code crashes (no checks for whether the game is over)
        pass
    plt.plot([game.step for game in history])
    plt.xlabel('Training episode')
    plt.ylabel('Final time step')
    plt.savefig('training_history_1.png')