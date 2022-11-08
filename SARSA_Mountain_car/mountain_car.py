import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim


class MountainCar:

    def __init__(self):
        self.x = np.linspace(-1.2, 0.6)
        self.car_pos = np.random.uniform(-0.6, -0.4)
        self.car_vel = 0
        self.step = 0

    @staticmethod
    def y(x):
        return np.cos(3*(x + np.pi/2))

    def is_state_final(self):
        if self.car_pos >= 0.6:
            return True
        if self.step >= 1000:
            return True
        return False

    def update_vel(self, F):
        self.car_vel += 0.001*F - 0.0025*np.cos(3*self.car_pos)
        if abs(self.car_vel) > 0.07:
            print('Too high velocity!')

    def update_pos(self):
        self.car_pos += self.car_vel
        self.car_pos = max(self.car_pos, -1.2)

    def make_action(self, action):
        self.update_vel(action)
        self.update_pos()
        self.step += 1

    def show_track(self, actor=None, filename=None):
        if actor is None:
            actor = lambda x: np.random.choice([-1, 0, 1]) # random actor if no function is supplied
        fig, ax = plt.subplots()
        ax.set(xlim=(-1.2, 0.6), ylim=(-1.5, 1.5))
        ax.plot(self.x, self.y(self.x), color='k')
        car = ax.plot(self.car_pos, self.y(self.car_pos), 'o', markersize=10)[0]

        def animate(i):
            self.make_action(actor(self))     # TODO: get action
            car.set_xdata(self.car_pos)
            car.set_ydata(self.y(self.car_pos))
            print(self.step)

        animation = anim.FuncAnimation(fig, animate, interval=100, frames=1000)
        if filename is not None:
            animation.save(filename)
        plt.show()


if __name__ == '__main__':
    t = MountainCar()
    t.show_track()
