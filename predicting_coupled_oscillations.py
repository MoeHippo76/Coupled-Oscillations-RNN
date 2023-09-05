from scipy.integrate import RK45
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# State = [position, velocity]
def SHM(t,state):
    state = state.reshape(2,2)
    k = 1
    # Function returns dState/dt
    return np.array([state[1,:], -k*state[0,:]])

def SolveSHM(initial_state, timesteps = 1000):
    SHM_solver = RK45(SHM, 0, initial_state, np.inf, max_step= 1e-1, vectorized= True) 

    positions = np.zeros((timesteps, 2))
    velocities = np.zeros((timesteps, 2))
    times = np.zeros(timesteps)
    for i in range(timesteps):
        times[i] = SHM_solver.t
        X, Y, V_x, V_y = SHM_solver.y
        positions[i] = np.array([X,Y])
        velocities[i] = np.array([V_x, V_y])
        SHM_solver.step()
        

    plt.plot(positions[:,0], positions[:,1], color = 'black', label = 'postion')
    #plt.plot(times, velocities, color = "hotpink", label = 'velocity')
    plt.legend()
    plt.title('Trajectory of 1D Simple Harmonic Oscillatior')
    plt.show()


def Solve(f, initial_state, timesteps = 60, final_time = np.inf):
    state_shape = np.shape(initial_state)
    
    if final_time == np.inf:
        step = 1e-1
    else:
        step = final_time/timesteps

    solver = RK45(f, 0, initial_state.flatten(), final_time, step, vectorized= True)
    
    ts = np.zeros(timesteps)
    States = np.zeros(np.shape(initial_state) + (timesteps,))
    
    for i in range(timesteps):
        ts[i] = solver.t
        State = np.array(solver.y).reshape(state_shape)
        States[...,i] = State
        solver.step()
    
    return ts, States

def multiple_coupled_oscillators(t, State):
    global spring_constants_k, coupling_constants_D , masses_m, Nparticles
    
    State = State.reshape(2, Nparticles)
    positions = State[0]
    velocities = State[1]

    #Forces on all the lighter particles (j>0)
    D_m = (coupling_constants_D/ masses_m) 
    accelerations = (-spring_constants_k / masses_m)  * positions + D_m * (positions[0] - positions)

    #Forces on the heavier particle (j = 0)
    summation = np.sum(coupling_constants_D * (positions - positions[0]), axis = 0)
    accelerations[0] = (-spring_constants_k[0] * positions[0]  + summation) / masses_m[0]

    return np.array([velocities, accelerations])

def test_1batch():
    spring_constants_k=np.array([1.0,1.3,0.7]) # the spring constants
    coupling_constants_D = 0.1 # the coupling between j=0 and the rest
    masses_m = np.array([2.0,1.0,1.0]) # the masses
    Nparticles = 3


    X0 = np.random.randn(2,3) # first index: x vs. v / second index: particles

    ts, Xs = Solve(multiple_coupled_oscillators, X0, 1000)


    for n in range(3):
        if n==0:
            color="black" # our 'heavy' particle!
            alpha=1
        else:
            color=None
            alpha=0.5
        plt.plot(ts,Xs[0,n],color=color,alpha=alpha)
    plt.show()

def multiple_coupled_oscillators_parallel(t, State):
    global spring_constants_k, coupling_constants_D , masses_m, number_particles, batchsize
    
    State = State.reshape(2, number_particles, batchsize)
    positions = State[0]
    velocities = State[1]

    D_m = (coupling_constants_D/ masses_m)[:,None] 
    accelerations = (-spring_constants_k / masses_m)[:, None]  * positions + D_m * (positions[0] - positions)
    summation = np.sum(coupling_constants_D[:,None] * (positions - positions[0]), axis = 0)
    accelerations[0] = (-spring_constants_k[0] * positions[0]  + summation) / masses_m[0]
    
    return np.array([velocities, accelerations])

def test_multiple_batches():
    number_particles = 6

    spring_constants_k = np.abs( 0.2 * np.random.randn(number_particles) + 1.0 ) # the spring constants
    coupling_constants_D = np.full(number_particles, 0.2) # the coupling between j=0 and the rest
    masses_m = np.abs( 0.2 * np.random.randn(number_particles) + 1.0 ) # the masses

    masses_m[0] = 3.0 # heavy particle
    spring_constants_k[0] = 3.0 # hard spring: make it still resonant with the rest!

    batchsize = 5
    X0 = np.random.randn(2, number_particles, batchsize) # first index: x vs. v / second index: particles / third: batch

    ts, Xs = Solve(multiple_coupled_oscillators_parallel, X0, 500)

    fig,ax=plt.subplots(ncols=batchsize,nrows=1,figsize=(batchsize*2,2))
    for j in range(batchsize):
        for n in range(number_particles):
            if n==0:
                color="black" # our 'heavy' particle!
                alpha=1
            else:
                color=None
                alpha=0.3
            ax[j].plot(ts,Xs[0,n,j],color=color,alpha=alpha)
            ax[j].axis('off')
    plt.show()


def init_LSTM():
    global net
    net = keras.Sequential()

    net.add(keras.layers.LSTM(20, input_shape = (None,  1),  return_sequences = True))

    net.add(keras.layers.Dense(1, activation= 'linear'))

    net.compile(loss = 'mean_squared_error', optimizer = 'adam')

def Predicting_particle1():
    number_particles = 3
    spring_constants_k = np.array([0.1, 0.0, 0.3]) # the spring constants
    coupling_constants_D = np.array([0.0,0.1,1.0]) # the coupling between j=0 and the rest
    masses_m = np.array([2.0,1.0,1.0])

    batchsize = 1
    X0 = np.random.randn(2, number_particles, 1)
    ts, Xs = Solve(multiple_coupled_oscillators_parallel, X0, 60, 30)
    plt.plot(ts, Xs[0,0,0])
    plt.plot(ts, Xs[0,1,0])
    plt.plot(ts, Xs[0,2,0])
    plt.show()

    init_LSTM()

    training_steps = 1000
    batchsize = 50
    costs = np.zeros(training_steps)

    for i in range(training_steps):
        X0 = np.random.randn(2, number_particles, batchsize)
        ts, Xs = Solve(multiple_coupled_oscillators_parallel, X0, 60, 30)
        costs[i] = net.train_on_batch(Xs[0,0,:][:,:, None] , Xs[0,1,:][:,:,None])

    plt.plot(costs)
    plt.show()


    num_tests = int(0.1 * batchsize)
    batchsize = num_tests
    X0 = np.random.randn(2, number_particles, num_tests)
    ts, Xs = Solve(multiple_coupled_oscillators_parallel, X0, 60, 30)
    test_inputs = Xs[0, 0, :][:, :, None]
    test_correct_outputs = Xs[0, 1, :][:, :, None]
    outputs = net.predict_on_batch(test_inputs)

    for i in range(num_tests):
        plt.plot(ts, test_correct_outputs[i], color = 'black')
        plt.plot(ts, outputs[i], color = 'hotpink')
        plt.show()

number_particles = 2
spring_constants_k = np.array([0.1, 0.0]) # the spring constants
coupling_constants_D = np.array([0.0,0.2]) # the coupling between j=0 and the rest
masses_m = np.array([2.0,1.0])

batchsize = 1
X0 = np.random.randn(2, number_particles, 1)
ts, Xs = Solve(multiple_coupled_oscillators_parallel, X0, 60, 30)
plt.plot(ts, Xs[0,0,0])
plt.plot(ts, Xs[0,1,0])
plt.show()

init_LSTM()
training_steps = 1000
batchsize = 50
costs = np.zeros(training_steps)
for i in range(training_steps):
    X0 = np.random.randn(2, number_particles, batchsize)
    ts, Xs = Solve(multiple_coupled_oscillators_parallel, X0, 60, 30)
    input = np.copy(Xs[0,0,:][:,:, None])
    input[:, 30:, 0] = 0 
    costs[i] = net.train_on_batch(input , Xs[0,0,:][:,:,None])
plt.plot(costs)
plt.show()

num_tests = int(0.1 * batchsize)
batchsize = num_tests
X0 = np.random.randn(2, number_particles, num_tests)
ts, Xs = Solve(multiple_coupled_oscillators_parallel, X0, 60, 30)
test_inputs = np.copy(Xs[0, 0, :][:, :, None])
test_inputs[:, 30:, 0] = 0
test_correct_outputs = Xs[0, 0, :][:, :, None]
outputs = net.predict_on_batch(test_inputs)
for i in range(num_tests):
    plt.plot(ts, test_inputs[i], color = 'black')
    plt.plot(ts, test_correct_outputs[i], color = 'red')
    plt.plot(ts, outputs[i], color = 'hotpink')
    plt.show()