import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

plt.ion()
# particles = 10_000
# particles_locations = np.array([0])
# time_steps = [50, 500]

# for time in time_steps:
#     for particle in len(particles):
#         move = random.randint(1,2)
#         if move == 1:
#             particles_locations = particles_locations + 0.1

#         else:
#             particles_locations = particles_locations - 0.1
        
#         plt.figure()
#         plt.plot(particles_locations)
#         plt.show()


particles = 10_000
particles_locations = np.zeros(particles)
time_steps = [50, 500]

for time in range(1, max(time_steps) + 1 ):
    steps = np.random.choice([-0.1, 0.1], size = particles)
    particles_locations += steps

    if time in time_steps:
        plt.figure(figsize = (10,8))
        plt.hist(particles_locations, bins = 100, density = True, alpha = 0.5, label = f'{time} time steps')
        plt.xlabel('position')
        plt.ylabel('Number of particles')
        plt.title('Random walk of particles')
        plt.legend()
        plt.show()



input('input to exit')