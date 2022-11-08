from abc import abstractmethod, ABC
from collections import defaultdict
from random import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

class Critic(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def new_episode(self):
        pass
    
    @abstractmethod
    def get_td_error(self, action, prev_state, game):
        pass
    
    @abstractmethod
    def renew_eligibility_trace(self, state):
        pass
    
    @abstractmethod
    def learn(self, td_error):
        pass


class LookupCritic(Critic):
    def __init__(self, learning_rate, discount_factor, eligibility_decay_rate):
        # python dictionary between states and values
        # these should be initialized to small, random values.
        self.values = defaultdict(random)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eligibility_decay_rate = eligibility_decay_rate
    
    def new_episode(self):
        self.eligibility_traces = {}
    
    def get_td_error(self, action, prev_state, game):
        # no real reinforcement applicable in this project
        new_state = game.get_state_id()
        if game.is_state_final():
            self.values[new_state] = game.get_score()
        return self.discount_factor * self.values[new_state] - self.values[prev_state]
    
    def renew_eligibility_trace(self, state):
        self.eligibility_traces[state] = 1
    
    def learn(self, td_error):
        for state, trace in self.eligibility_traces.items():
            self.values[state] += self.learning_rate * trace * td_error
            self.eligibility_traces[state] *= self.discount_factor * self.eligibility_decay_rate


class DeepLearningCritic(Critic):
    def __init__(self, learning_rate, discount_factor, eligibility_decay_rate, layer_sizes):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eligibility_decay_rate = eligibility_decay_rate
        self.layer_sizes = layer_sizes
        self.eligibility_traces = 0
        network = Sequential()
        if len(layer_sizes) < 1:
            raise ValueError('Deep learning critic needs at least 1 layer specified (input)')
        network.add(Input(self.layer_sizes[0]))
        for layer_size in layer_sizes[1:]:
            network.add(Dense(layer_size, activation='swish'))
        network.add(Dense(1, activation='linear'))
        network.compile(optimizer=SGD(learning_rate=learning_rate), loss=MeanSquaredError())
        self.model = network
    
    def get_inputs(self, game_id):
        input_length = self.layer_sizes[0]
        binary_string = format(game_id, f'0{input_length}b') # binary string, with leading zeroes, length equal to input layer
        binary_array = [int(c) for c in binary_string]
        return binary_array
    
    def new_episode(self):
        self.eligibility_traces = None
    
    def get_td_error(self, action, prev_state, game):
        prev_state_input = self.get_inputs(prev_state)
        estimated_value = self.model.predict([prev_state_input])[0][0]
        if game.is_state_final():
            new_state_value = game.get_score()
        else:
            new_state_input = self.get_inputs(game.get_state_id())
            new_state_value = self.model.predict([new_state_input])[0][0]
        actual_value = self.discount_factor * new_state_value
        return actual_value - estimated_value
    
    def renew_eligibility_trace(self, state):
        input_vector = self.get_inputs(state)
        input_tensor = tf.convert_to_tensor([input_vector])
        with tf.GradientTape() as tape:
            output = self.model(input_tensor)
        gradient = tape.gradient(output, self.model.trainable_weights)
        if self.eligibility_traces == None:
            self.eligibility_traces = gradient
        else:
            self.eligibility_traces = [
                self.eligibility_decay_rate * prev_elig + new_grad
                for prev_elig, new_grad in zip(self.eligibility_traces, gradient)]

    def learn(self, td_error):
        if self.eligibility_traces == None:
            return
        scaled_grad = [-trace * td_error for trace in self.eligibility_traces]
        self.model.optimizer.apply_gradients(zip(scaled_grad, self.model.trainable_weights))