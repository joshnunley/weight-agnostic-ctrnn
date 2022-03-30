import numpy as np
import matplotlib.pyplot as plt

from CartPoleEnv import CartPoleEnv

from CTRNN import CTRNN
from EvolSearch import EvolSearch


ctrnn_size = 40
num_inputs = 4
pop_size = 200
step_size = 0.01


def map_state_to_f(states, force_mult):
    pos_force = np.sum(states[: int(ctrnn_size / 2)] > 0.5)
    neg_force = np.sum(states[int(ctrnn_size / 2) :] > 0.5)

    f = force_mult * (pos_force - neg_force)

    return f


def cart_pole_fitness(ctrnn_parameters, runs=5, fitness_type="mean"):
    ctrnn = CTRNN(size=ctrnn_size, step_size=step_size)

    num_weights = ctrnn_size**2

    ctrnn.weights = (
        10 * ctrnn_parameters[:num_weights].reshape((ctrnn_size, ctrnn.size)) - 5
    )
    ctrnn.taus = ctrnn_parameters[num_weights : (num_weights + ctrnn_size)] + 0.0001
    ctrnn.biases = (
        10
        * ctrnn_parameters[(num_weights + ctrnn_size) : (num_weights + 2 * ctrnn_size)]
        - 5
    )

    force_mult = ctrnn_parameters[(num_weights + 2 * ctrnn_size)] + 1 / ctrnn_size

    input_weights = 2 * ctrnn_parameters[(num_weights + 2 * ctrnn_size) + 1 :] - 1
    if num_inputs > 1:
        input_weights = input_weights.reshape((ctrnn_size, num_inputs))

    rewards = np.zeros(runs)
    for i in range(runs):
        cart_pole = CartPoleEnv()
        cart_pole.reset()
        cart_pole_state = cart_pole.state

        ctrnn.states = np.zeros(ctrnn_size)
        ctrnn_input = np.zeros(ctrnn_size)

        for _ in range(1000):
            ctrnn_input = input_weights.dot(cart_pole_state)
            for __ in range(int(cart_pole.tau / ctrnn.step_size)):
                ctrnn.euler_step(ctrnn_input)

            states = ctrnn.outputs
            if np.isnan(states[-1]):
                break
            f = map_state_to_f(states, force_mult)

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

genotype_size = ctrnn_size**2 + 2 * ctrnn_size + 1 + num_inputs * ctrnn_size

evol_params = {
    "num_processes": 6,
    "pop_size": pop_size,  # population size
    "genotype_size": genotype_size,  # dimensionality of solution
    "fitness_function": cart_pole_fitness,  # custom function defined to evaluate fitness of a solution
    "elitist_fraction": 0.1,  # fraction of population retained as is between generations
    "mutation_variance": 0.005,  # mutation noise added to offspring.
}
initial_pop = np.random.uniform(size=(pop_size, genotype_size))
evolution = EvolSearch(evol_params, initial_pop)
evolution.step_generation()

best_fitness = []
best_fitness.append(evolution.get_best_individual_fitness())
print(best_fitness)

print(evolution.get_best_individual())

mean_fitness = []
mean_fitness.append(evolution.get_mean_fitness())

while best_fitness[-1] < 60:
    evolution.step_generation()
    best_individual = evolution.get_best_individual()
    best_individual_fitness = cart_pole_fitness(best_individual, 10, "mean")

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

num_weights = ctrnn_size**2

ctrnn.weights = (
    10 * ctrnn_parameters[:num_weights].reshape((ctrnn_size, ctrnn.size)) - 5
)
ctrnn.taus = ctrnn_parameters[num_weights : (num_weights + ctrnn_size)] + 0.0001
ctrnn.biases = (
    10 * ctrnn_parameters[(num_weights + ctrnn_size) : (num_weights + 2 * ctrnn_size)]
    - 5
)

force_mult = ctrnn_parameters[(num_weights + 2 * ctrnn_size)] + 1 / ctrnn_size

input_weights = 2 * ctrnn_parameters[(num_weights + 2 * ctrnn_size) + 1 :] - 1
if num_inputs > 1:
    input_weights = input_weights.reshape((ctrnn_size, num_inputs))

print(ctrnn.weights, ctrnn.taus, ctrnn.biases, force_mult)
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

        output_state1[ind] = ctrnn.states[-1]
        output_state2[ind] = ctrnn.states[-2]
        output_state3[ind] = ctrnn.states[-3]
        output_state4[ind] = ctrnn.states[-4]
        ind += 1

    states = ctrnn.outputs
    if np.isnan(states[-1]):
        break
    f = map_state_to_f(states, force_mult)
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
