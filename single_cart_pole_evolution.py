import numpy as np
import matplotlib.pyplot as plt

from CartPoleEnv import CartPoleEnv

from CTRNN import CTRNN
from EvolSearch import EvolSearch


ctrnn_size = 4
pop_size = 200
step_size = 0.005


def map_state_to_f(signal1, force_mult_scale, force_mult):
    return force_mult_scale * force_mult * signal1


def cart_pole_fitness(ctrnn_parameters, runs=5, fitness_type="mean"):
    ctrnn = CTRNN(size=ctrnn_size, step_size=step_size)

    num_weights = ctrnn_size ** 2

    ctrnn.weights = ctrnn_parameters[:num_weights].reshape((ctrnn_size, ctrnn.size))
    ctrnn.taus = ctrnn_parameters[num_weights : (num_weights + ctrnn_size)]
    ctrnn.biases = ctrnn_parameters[
        (num_weights + ctrnn_size) : (num_weights + 2 * ctrnn_size)
    ]

    force_mult = ctrnn_parameters[-1]
    force_mult_scale = ctrnn_parameters[-2]

    total_reward = 0
    rewards = np.zeros(runs)
    for i in range(runs):
        cart_pole = CartPoleEnv()
        cart_pole.reset()
        cart_pole_state = cart_pole.state

        ctrnn.states = np.zeros(ctrnn_size)
        ctrnn_input = np.zeros(ctrnn_size)

        for _ in range(1000):
            ctrnn_input[: cart_pole_state.shape[0]] = cart_pole_state
            for __ in range(int(cart_pole.tau / ctrnn.step_size)):
                ctrnn.euler_step(ctrnn_input)

            states = ctrnn.states
            if np.isnan(states[-1]):
                break
            f = map_state_to_f(states[-1], force_mult_scale, force_mult)

            cart_pole_state, reward, done, d = cart_pole.step(f)

            if done:
                break
            rewards[i] += reward
    if fitness_type == "mean":
        return np.mean(rewards)
    elif fitness_type == "min":
        return np.min(rewards)


########################
# Evolve Solutions
########################

genotype_size = ctrnn_size ** 2 + 2 * ctrnn_size + 2

evol_params = {
    "num_processes": 6,
    "pop_size": pop_size,  # population size
    "genotype_size": genotype_size,  # dimensionality of solution
    "fitness_function": cart_pole_fitness,  # custom function defined to evaluate fitness of a solution
    "elitist_fraction": 0.1,  # fraction of population retained as is between generations
    "mutation_variance": 0.05,  # mutation noise added to offspring.
}
initial_pop = np.zeros(shape=(pop_size, genotype_size))
variable_mins = []
variable_maxes = []
weight_lims = {"min": -1, "max": 1}
tau_lims = {"min": 0.00001, "max": 1}
bias_lims = {"min": -1, "max": 1}
force_mult_scale_lims = {"min": 900, "max": 1000}
force_mult_lims = {"min": 0.7, "max": 1}
for i in range(pop_size):
    for j in range(genotype_size):
        if j < ctrnn_size ** 2:
            if i == 0:
                variable_mins.append(weight_lims["min"])
                variable_maxes.append(weight_lims["max"])
        elif j < ctrnn_size ** 2 + ctrnn_size:
            if i == 0:
                variable_mins.append(tau_lims["min"])
                variable_maxes.append(tau_lims["max"])
        elif j < ctrnn_size ** 2 + 2 * ctrnn_size:
            if i == 0:
                variable_mins.append(bias_lims["min"])
                variable_maxes.append(bias_lims["max"])
        elif j == ctrnn_size ** 2 + 2 * ctrnn_size:
            if i == 0:
                variable_mins.append(force_mult_scale_lims["min"])
                variable_maxes.append(force_mult_scale_lims["max"])
        elif j == ctrnn_size ** 2 + 2 * ctrnn_size + 1:
            if i == 0:
                variable_mins.append(force_mult_lims["min"])
                variable_maxes.append(force_mult_lims["max"])
        initial_pop[i, j] = np.random.uniform(
            low=variable_mins[j],
            high=variable_maxes[j],
        )
# Initialize with center crossing
first_bias_ind = ctrnn_size ** 2 + ctrnn_size
last_bias_ind = ctrnn_size ** 2 + 2 * ctrnn_size
weights = initial_pop[:, : ctrnn_size ** 2].reshape((pop_size, ctrnn_size, ctrnn_size))
for i in range(pop_size):
    for j in range(first_bias_ind, last_bias_ind):
        bias_neuron_ind = j - ctrnn_size ** 2 - ctrnn_size
        initial_pop[i, j] = -np.sum(weights[i, :, bias_neuron_ind]) / 2
        # initial_pop[i, j] = -np.sum(weights[i, bias_neuron_ind, :]) / 2

good_parameters = np.array(
    [
        0.81691084,
        0.87417405,
        -0.73922413,
        -1.0,
        0.9042886,
        0.54576909,
        -0.21203265,
        -0.70939197,
        0.58833687,
        -0.72307045,
        0.22894309,
        -0.65086737,
        -0.10981333,
        1.0,
        1.0,
        -0.95540368,
        0.47420226,
        0.00914973,
        0.50167858,
        0.34367007,
        -0.85881434,
        -0.90400767,
        -0.15560706,
        0.48952325,
        500,
        0.9603362,
    ]
)

good_parameters2 = np.array(
    [
        0.89410088,
        0.82467332,
        -0.27765919,
        -0.85824984,
        1.0,
        0.36092845,
        -0.05562215,
        -0.17103638,
        0.98595278,
        -0.67007787,
        0.74679536,
        -0.78121391,
        -0.20551598,
        0.84023588,
        1.0,
        -0.98951772,
        1.0,
        0.00696009,
        0.39638457,
        0.22886115,
        -0.19656279,
        -0.93734141,
        -0.19850529,
        0.75164833,
        1000,
        0.82777879,
    ]
)
# Initialize with previously found parameters
if False:
    initial_pop[-1, :] = good_parameters
    initial_pop[-2, :] = good_parameters2


evolution = EvolSearch(evol_params, initial_pop, variable_mins, variable_maxes)
evolution.step_generation()

best_fitness = []
best_fitness.append(evolution.get_best_individual_fitness())
print(best_fitness)

print(evolution.get_best_individual())

mean_fitness = []
mean_fitness.append(evolution.get_mean_fitness())

while best_fitness[-1] < 600:
    evolution.step_generation()
    best_individual = evolution.get_best_individual()
    best_individual_fitness = cart_pole_fitness(best_individual, 50, "mean")

    best_fitness.append(best_individual_fitness)
    mean_fitness.append(evolution.get_mean_fitness())

    print(len(best_fitness), best_fitness[-1], evolution.get_mean_fitness())

#########################
# Run Simulation
#########################
ctrnn_parameters = evolution.get_best_individual()
print(ctrnn_parameters)
print(cart_pole_fitness(ctrnn_parameters))
print(cart_pole_fitness(ctrnn_parameters))
print(cart_pole_fitness(ctrnn_parameters))

ctrnn = CTRNN(size=ctrnn_size, step_size=step_size)

num_weights = ctrnn_size ** 2

ctrnn.weights = ctrnn_parameters[:num_weights].reshape((ctrnn_size, ctrnn_size))
ctrnn.taus = ctrnn_parameters[num_weights : (num_weights + ctrnn_size)]
ctrnn.biases = ctrnn_parameters[
    (num_weights + ctrnn_size) : (num_weights + 2 * ctrnn_size)
]

force_mult = ctrnn_parameters[-1]
force_mult_scale = ctrnn_parameters[-2]
print(ctrnn.weights, ctrnn.taus, ctrnn.biases, force_mult_scale, force_mult)

cart_pole = CartPoleEnv(tau=0.02)

cart_pole.reset()
ctrnn_input = np.zeros(ctrnn_size)
ctrnn.states = np.zeros(ctrnn_size)
cart_pole_state = cart_pole.state

steps = 5000
output_state1 = np.zeros(int(cart_pole.tau / ctrnn.step_size) * steps)
output_state2 = np.zeros(int(cart_pole.tau / ctrnn.step_size) * steps)
output_state3 = np.zeros(int(cart_pole.tau / ctrnn.step_size) * steps)
output_state4 = np.zeros(int(cart_pole.tau / ctrnn.step_size) * steps)
reward_total = 0

ind = 0
for i in range(steps):
    ctrnn_input[: cart_pole_state.shape[0]] = cart_pole_state
    for _ in range(int(cart_pole.tau / ctrnn.step_size)):
        ctrnn.euler_step(ctrnn_input)

        states = ctrnn.states
        output_state1[ind] = states[-1]
        output_state2[ind] = states[-2]
        output_state3[ind] = states[-3]
        output_state4[ind] = states[-4]
        ind += 1

    states = ctrnn.states
    if np.isnan(states[-1]):
        break
    f = map_state_to_f(states[-1], force_mult_scale, force_mult)
    cart_pole.render()
    cart_pole_state, reward, done, d = cart_pole.step(f)

    reward_total += reward

print(reward_total)

cart_pole.close()

plt.figure(0)
plt.plot(best_fitness)

plt.figure(1)
plt.plot(mean_fitness)

plt.figure(2)
plt.plot(output_state1)

plt.figure(3)
plt.plot(output_state2)

plt.figure(4)
plt.plot(output_state3)

plt.figure(5)
plt.plot(output_state4)

plt.show()
