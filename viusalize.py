from matplotlib import pyplot as plt
import os

class goal_plot:

    def __init__(self, dir):

        self.goal_list = []
        self.subgoal_list = []
        self.dir = dir

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 10))

        self.ax1.set_title('the distribution of goals')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')

        self.ax2.set_title('the distribution of sub goals')
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')

        self.ax1.set_xlim([0.5, 2])
        self.ax1.set_ylim([0, 1.5])

        self.ax2.set_xlim([0.5, 2])
        self.ax2.set_ylim([0, 1.5])

    def add_goal(self, g):
        self.ax1.scatter(g[0], g[1], s=2, c='r', marker='.', alpha=0.1)
        plt.savefig(os.path.join(self.dir, 'goal.png'))

    def add_subgoal(self, g):
        self.ax2.scatter(g[0], g[1], s=2, c='r', marker='.', alpha=0.1)
        plt.savefig(os.path.join(self.dir, 'goal.png'))