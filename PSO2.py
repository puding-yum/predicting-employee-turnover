import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


class PSO:
    def __init__(self, fit_func, num_dim=30, no_particles=10, no_epochs=100):
        self.no_particles = no_particles   # number of particles
        self.no_epochs = no_epochs     # number of epochs
        self.c1 = .5     # cognitive part constant
        self.c2 = .5     # social part constant
        self.particles = []  # store objects of each paricle
        self.vmin,vmax = -4,4    # min and max velocity of particle
        self.w = .9  # inertia of particle
        self.fitness_overall = []
        self.pos_overall = []

    def run(self, obj, dim, algo=1):
        for epoch in range(0,self.no_epochs):
            if epoch == 0:
                initialize(obj,dim,algo)
                best_fitness = fitness_overall[-1]
                gbest_pos = pos_overall[-1]
                print("Iteration : ",epoch+1," -> ",best_fitness,end='\t')
                print(gbest_pos)
            else:
                
                cal(obj,dim,algo,best_fitness,gbest_pos)
                best_fitness = fitness_overall[-1]
                gbest_pos = pos_overall[-1]
                print("Iteration : ",epoch+1," -> ",best_fitness,end='\t')
                print(gbest_pos)

        x = np.arange(0,no_epochs)
        y = np.array(fitness_overall)
        #plt.scatter(x,y)
        plt.ylim((50,110))
        plt.plot(x,y)
        plt.show()