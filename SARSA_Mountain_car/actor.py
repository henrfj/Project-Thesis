from random import random, randint

class Actor:
    def __init__(self, learning_rate, discount_factor, eligibility_decay_rate, epsilon, epsilon_decay_rate):
        # maps state-action pairs to values that indicate desirability
        self.policy = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eligibility_decay_rate = eligibility_decay_rate
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
    
    def new_episode(self):
        self.eligibility_traces = {}
        self.epsilon *= self.epsilon_decay_rate
    
    def act(self, game, exploit):
        state = game.get_state_id()
        if game.is_state_final():
            return None
        if state not in self.policy:
            self.policy[state] = {}
            for action in game.get_possible_moves():
                self.policy[state][action] = 0
        if exploit or random() >= self.epsilon:
            best_action = None
            best_value = -float('inf')
            for action in self.policy[state]:
                if not self.policy[state][action] > -float('inf'):
                    print('wtf')
                    print(self.policy[state][action])
                    exit()
                if self.policy[state][action] > best_value:
                    best_action = action
                    best_value = self.policy[state][action]
            # return the most favorable action
            return best_action
        # return a random action
        actions = list(self.policy[state].keys())
        action = actions[randint(0, len(actions) - 1)]
        return action
    
    def renew_eligibility_trace(self, state, action):
        self.eligibility_traces[(state, action)] = 1

    def learn(self, td_error):
        for state_action_pair, trace in self.eligibility_traces.items():
            state, action = state_action_pair
            self.policy[state][action] += self.learning_rate * trace * td_error
            self.eligibility_traces[state_action_pair] *= self.discount_factor * self.eligibility_decay_rate