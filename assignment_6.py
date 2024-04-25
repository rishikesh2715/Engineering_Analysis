import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def generate_poisson_arrivals(rate, max_time):
    arrivals = []
    current_time = 0
    while current_time < max_time:
        inter_arrival_time = np.random.exponential(1/rate)
        current_time += inter_arrival_time
        if current_time < max_time:
            arrivals.append(current_time)
    return arrivals

def generate_constant_rate_arrivals(rate, max_time):
    interval = rate
    return np.arange(interval, max_time, interval)

def simulate_system(max_time, poisson_rate, constant_rate, failure_rate, down_time):
    poisson_arrivals = generate_poisson_arrivals(poisson_rate, max_time)
    constant_arrivals = generate_constant_rate_arrivals(constant_rate, max_time)
    next_failure_time = np.random.exponential(1/failure_rate)
    
    assembled_count = 0
    current_time = 0
    failure_active_until = -1

    events = sorted([(t, 'poisson') for t in poisson_arrivals] + [(t, 'constant') for t in constant_arrivals])

    for event_time, event_type in events:
        if event_time >= next_failure_time:
            failure_active_until = event_time + down_time
            next_failure_time += np.random.exponential(1/failure_rate)

        if event_time >= failure_active_until:
            assembled_count += 1

    return assembled_count, assembled_count / max_time

def generate_time_to_failure(failure_rate):
    U = np.random.uniform()
    T = -np.log(U) / failure_rate
    return T


failure_rate = 0.01  # failures per second
time_to_failure = generate_time_to_failure(failure_rate)
print(f"Time to next failure: {time_to_failure} seconds")

def plot_poisson(mu, max_time, num_samples):
    samples = [len(generate_poisson_arrivals(mu, max_time)) for _ in range(num_samples)]
    
    # Theoretical PMF based on Poisson distribution
    max_val = max(samples)
    x = np.arange(0, max_val + 1)
    y = poisson.pmf(x, mu * max_time)
    
    
    # Plot for Simulated Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=np.arange(-0.5, max(samples)+1), density=True, alpha=0.7, color='blue', edgecolor='black', linewidth=1.2)
    plt.title(f'Histogram of Random Numbers with Poisson Distribution (Mean = {mu})')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

    # Plot for Analytical PMF
    plt.figure(figsize=(10, 6))
    plt.stem(x, y, 'b', markerfmt='bo', basefmt="r-", use_line_collection=True, label='Analytic Poisson PMF')
    plt.title(f'Analytical Poisson PMF for mu={mu}')
    plt.xlabel('Number of Arrivals')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

# Simulation parameters
max_time = 10000  # seconds
poisson_rate = 1  # parts per second
constant_rate = 1.5  # time in seconds per part
failure_rate = 0.01  # failures per second
down_time = 5  # seconds

# Run simulation
output_count, average_rate = simulate_system(max_time, poisson_rate, constant_rate, failure_rate, down_time)
print(f"Total assembled parts: {output_count}")
print(f"Average output rate: {average_rate} parts per second")


# Parameters
mu_values = [6, 15]
max_time = 1  # time interval to count arrivals
num_samples = 10000  # number of samples to generate for the histograms

for mu in mu_values:
    plot_poisson(mu, max_time, num_samples)