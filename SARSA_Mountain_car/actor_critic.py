from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import animation as anim
    
class ActorCritic:
    def __init__(self, game, actor, critic):
        self.base_game = game # should never be manipulated directly, only by deepcopy
        self.actor = actor
        self.critic = critic
        self.history = [] # keeps track of how well the model does over time
    
    def run_episode(self, exploit=False, graphics=False, graphics_delay_ms=200):
        game = deepcopy(self.base_game)
        self.actor.new_episode()
        self.critic.new_episode()
        # set some object attributes in order to use the outer scope in the function
        self._state = game.get_state_id()
        self._action = self.actor.act(game, exploit)
        def make_move():
            state, action = self._state, self._action
            game.move(action)
            next_state = game.get_state_id()
            next_action = self.actor.act(game, exploit)
            td_error = self.critic.get_td_error(action, state, game)
            self.actor.renew_eligibility_trace(state, action)
            self.critic.renew_eligibility_trace(state)
            self.actor.learn(td_error)
            self.critic.learn(td_error)
            self._state = next_state
            self._action = next_action
        
        def graphics_frame(i):
            if i == 0:
                return game.show_game()
            if not game.is_state_final():
                make_move()
            return game.show_game()
        if graphics:
            figure = plt.gcf()
            animation = anim.FuncAnimation(figure, graphics_frame, interval=graphics_delay_ms,
                                           frames=game.grid.size ** 2, blit=True, repeat=False)
            plt.show()
        while not game.is_state_final():
            make_move()
        del self._state
        del self._action
        return game.get_peg_count()
    
    def run_episodes(self, iterations, **kwargs):
        for i in range(iterations):
            print(f'Episode {i+1}/{iterations}')
            self.history.append(self.run_episode(**kwargs))
    
    def plot_history(self, block=True):
        plt.plot(self.history)
        plt.show(block=block)
