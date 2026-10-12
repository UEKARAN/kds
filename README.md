## PRACTICAL 1A
**AIM:** Design a simple linear neural network model.

**CODE:**
```python
x = float(input("Enter Value for X:"))
w = float(input("Enter Value for W:"))
b = float(input("Enter Value for B:"))
y = (w * x) + b
print(f"Output y = {y}")
```

---

## PRACTICAL 1B
**AIM:** Calculate the output of neural net using both binary and bipolar sigmoidal function.

**CODE:**
```python
import math
x1 = float(input("X1="))
x2 = float(input("X2="))
x3 = float(input("X3="))
w1 = float(input("W1="))
w2 = float(input("W2="))
w3 = float(input("W3="))
yin = (x1*w1) + (x2*w2) + (x3*w3)
print("Net input is: ", round(yin, 4))
binary = 1/(1+math.exp(-yin))
print("Binary Sigmoid Output: ", round(binary, 4))
bipolar = (2/(1+math.exp(-yin)))-1
print("Bipolar Sigmoid Output:", round(bipolar, 4))
```

---

## PRACTICAL 2A
**AIM:** Generate AND/NOT function using McCulloch-Pitts neural NET.

**CODE:**
```python
import numpy

num_ip = int(input("Enter the number of inputs: "))
w1 = 1
w2 = 1
print("For the ", num_ip, " inputs calculate the net input using yin = x1w1 + x2w2 ")

x1 = []
x2 = []
for j in range(0, num_ip):
    ele1 = int(input("x1 ="))
    ele2 = int(input("x2 ="))
    x1.append(ele1)
    x2.append(ele2)

print("x1 = ", x1)
print("x2 = ", x2)

n = x1 * w1
m = x2 * w2

Yin = []
for i in range(0, num_ip):
    Yin.append(n[i] + m[i])
print("Yin = ", Yin)

Yin = []
for i in range(0, num_ip):
    Yin.append(n[i] - m[i])
print("After assuming one weight as excitory and the other as inhibitory yin =", Yin)

Y = []
for i in range(0, num_ip):
    if Yin[i] >= 1:
        ele = 1
        Y.append(ele)
    if Yin[i] < 1:
        ele = 0
        Y.append(ele)
print("Y = ", Y)
```

---

## PRACTICAL 2B
**AIM:** Generate XOR function using McCulloch-Pitts neural net.

**CODE:**
```python
import math
import numpy
import random

INPUT_NODES = 2
OUTPUT_NODES = 1
HIDDEN_NODES = 2
MAX_ITERATIONS = 130000
LEARNING_RATE = .2

print("Neural Network Program")

class network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.total_nodes = input_nodes + hidden_nodes + output_nodes
        self.learning_rate = learning_rate
        self.values = numpy.zeros(self.total_nodes)
        self.expectedValues = numpy.zeros(self.total_nodes)
        self.thresholds = numpy.zeros(self.total_nodes)
        self.weights = numpy.zeros((self.total_nodes, self.total_nodes))
        random.seed(10000)
        for i in range(self.input_nodes, self.total_nodes):
            self.thresholds[i] = random.random() / random.random()
            for j in range(i + 1, self.total_nodes):
                self.weights[i][j] = random.random() * 2

    def process(self):
        for i in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
            w_i = 0.0
            for j in range(self.input_nodes):
                w_i += self.weights[j][i] * self.values[j]
            w_i -= self.thresholds[i]
            self.values[i] = 1 / (1 + math.exp(-w_i))
        for i in range(self.input_nodes + self.hidden_nodes, self.total_nodes):
            w_i = 0.0
            for j in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
                w_i += self.weights[j][i] * self.values[j]
            w_i -= self.thresholds[i]
            self.values[i] = 1 / (1 + math.exp(-w_i))

    def processErrors(self):
        sumOfSquaredErrors = 0.0
        for i in range(self.input_nodes + self.hidden_nodes, self.total_nodes):
            error = self.expectedValues[i] - self.values[i]
            sumOfSquaredErrors += math.pow(error, 2)
            outputErrorGradient = self.values[i] * (1 - self.values[i]) * error
            for j in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
                delta = self.learning_rate * self.values[j] * outputErrorGradient
                self.weights[j][i] += delta
            self.thresholds[i] -= self.learning_rate * outputErrorGradient
            for k in range(self.input_nodes):
                hiddenErrorGradient = self.values[k] * self.values[i] * outputErrorGradient * self.weights[k][i]
                delta = self.learning_rate * self.values[k] * hiddenErrorGradient
                self.weights[k][i] += delta
            self.thresholds[i] -= self.learning_rate * hiddenErrorGradient
        return sumOfSquaredErrors

class sampleMaker:
    def __init__(self, network):
        self.counter = 0
        self.network = network

    def setXor(self, x):
        if x == 0:
            self.network.values[0] = 1
            self.network.values[1] = 1
            self.network.expectedValues[4] = 0
        elif x == 1:
            self.network.values[0] = 0
            self.network.values[1] = 1
            self.network.expectedValues[4] = 1
        elif x == 2:
            self.network.values[0] = 1
            self.network.values[1] = 0
            self.network.expectedValues[4] = 1
        else:
            self.network.values[0] = 0
            self.network.values[1] = 0
            self.network.expectedValues[4] = 0

    def setNextTrainingData(self):
        self.setXor(self.counter % 4)
        self.counter += 1

net = network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
samples = sampleMaker(net)

for i in range(MAX_ITERATIONS):
    samples.setNextTrainingData()
    net.process()
    error = net.processErrors()
    if i > (MAX_ITERATIONS - 5):
        output = (net.values[0], net.values[1], net.values[4], net.expectedValues[4], error)
        print(output)

print(net.weights)
print(net.thresholds)
```

---

## PRACTICAL 3A
**AIM:** Write a program to implement Hebb's rule.

**CODE:**
```python
import numpy as np

x1 = np.array([1,1,1,-1,1,-1,1,1,1,1])
x2 = np.array([1,1,1,-1,1,-1,1,1,1,1])
b = 0
y = np.array([1,-1])
wtold = np.zeros((9,))
wtnew = np.zeros((9,))
wtnew = wtnew.astype(int)
wtold = wtold.astype(int)
bais = 0

print("First input with target=1")
for i in range(0,9):
    wtold[i] = wtold[i] + x1[i]*y[0]
wtnew = wtold
b = b + y[0]

print("Second input with target=-1")
for i in range(0,9):
    wtnew[i] = wtold[i] + x2[i]*y[1]
b = b + y[1]

print("new wt=", wtnew)
print("Bias value", b)
```

---

## PRACTICAL 3B
**AIM:** Write a program to implement delta rule.

**CODE:**
```python
import numpy as np
import time

np.set_printoptions(precision=2)
x = np.zeros((3,))
weights = np.zeros((3,))
desired = np.zeros((3,))
actual = np.zeros((3,))

for i in range(0, 3):
    x[i] = float(input("Initial Inputs:"))
for i in range(0, 3):
    weights[i] = float(input("Initial weights:-"))
for i in range(0, 3):
    desired[i] = float(input("Desired output:-"))

a = float(input("Enter learning rate:-"))
actual = x * weights
print("Actual", actual)
print("Desired", desired)

while True:
    if np.array_equal(desired, actual):
        break
    else:
        for i in range(0, 3):
            weights[i] = weights[i] + a * (desired[i] - actual[i])
        actual = x * weights
        print("Weights", weights)
        print("Actual", actual)
        print("Desired", desired)
        print("*" * 30)

print("Final Output")
print("Corrected weights", weights)
print("Actual", actual)
print("Desired", desired)
```

---

## PRACTICAL 4A
**AIM:** Write a program for Back Propagation Algorithm

**CODE:**
```python
import numpy as np
import decimal
import math

np.set_printoptions(precision=2)
v1 = np.array([0.6, 0.3])
v2 = np.array([-0.1, 0.4])
w = np.array([-0.2, 0.4, 0.1])
b1 = 0.3
b2 = 0.5
x1 = 0
x2 = 1
alpha = 0.25

print("Calculate net input to z1 layer.")
zin1 = round(b1 + x1 * v1[0] + x2 * v2[0], 4)
print("zin1=", round(zin1, 3))

print("Calculate net input to z2 layer.")
zin2 = round(b2 + x1 * v1[1] + x2 * v2[1], 4)
print("zin2=", round(zin2, 4))

print("Apply activation function to calculate output.")
z1 = 1 / (1 + math.exp(-zin1))
z1 = round(z1, 4)
z2 = 1 / (1 + math.exp(-zin2))
z2 = round(z2, 4)
print("z1=", z1)
print("z2=", z2)

print("Calculate net input to output layer.")
yin = w[0] + z1 * w[1] + z2 * w[2]
print("yin=", yin)

print("Calculate net output.")
y = 1 / (1 + math.exp(-yin))
print("y=", y)

fyin = y * (1 - y)
dk = (1 - y) * fyin
print("dk=", dk)
dw1 = alpha * dk * z1
dw2 = alpha * dk * z2
dw0 = alpha * dk

print("Compute error portion in delta.")
din1 = dk * w[1]
din2 = dk * w[2]
print("din1=", din1)
print("din2=", din2)

print("Error in delta.")
fzin1 = z1 * (1 - z1)
print("fzin1=", fzin1)
d1 = din1 * fzin1
fzin2 = z2 * (1 - z2)
print("fzin2=", fzin2)
d2 = din2 * fzin2
print("d1=", d1)
print("d2=", d2)

print("Changes in weights between input and hidden layer.")
dv11 = alpha * d1 * x1
print("dv11=", dv11)
dv21 = alpha * d1 * x2
print("dv21=", dv21)
dv01 = alpha * d1
print("dv01=", dv01)
dv12 = alpha * d2 * x1
print("dv12=", dv12)
dv22 = alpha * d2 * x2
print("dv22=", dv22)
dv02 = alpha * d2
print("dv02=", dv02)

print("Final weights of network.")
v1 = v1.astype(float)
v2 = v2.astype(float)
w = w.astype(float)

v1[0] = v1[0] + dv11
v1[1] = v1[1] + dv12
print("v1=", v1)
v2[0] = v2[0] + dv21
v2[1] = v2[1] + dv22
print("v2=", v2)

w[1] = w[1] + dw1
w[2] = w[2] + dw2
b1 = b1 + dv01
b2 = b2 + dv02
w[0] = w[0] + dw0
print("w=", w)
print("bias b1=", b1, "b2=", b2)
```

---

## PRACTICAL 4B
**AIM:** Write a program for error Backpropagation algorithm.

**CODE:**
```python
import math

a0 = 1
t = 1
w10 = float(input("Enter First Weight:"))
b10 = float(input("Enter First Bias Weight:"))
w20 = float(input("Enter Second Weight:"))
b20 = float(input("Enter Second Bias Weight:"))
c = float(input("Enter Learning Coefficient:"))

n1 = float(w10*c + b10)
a1 = math.tanh(float(n1))
n2 = float(w20*a1 + b20)
a2 = math.tanh(float(n2))

e = t - a2
s2 = s2 * (1 - a2*a2) * e
s1 = (1 - a1 * a1) * w20 * s2
w21 = w20 - (c * s2 * a1)
w11 = w10 - (c * s1 * a0)
b21 = b20 - (c * s2)
b11 = b10 - (c * s1)

print("The updated weight of first n/w w11=", w11)
print("The updated weight of second n/w w21=", w21)
print("The updated base of first n/w b10=", b10)
print("The updated base of second n/w b20=", b20)
```

---

## PRACTICAL 5A
**AIM:** Write a program for Hopfield Network.

**CODE:**
```python
class Neuron:
    def __init__(self, j):
        self.activation = 0
        self.weightv = [0] * 4
        for i in range(4):
            self.weightv[i] = j[i]

    def act(self, m, x):
        a = 0
        for i in range(m):
            a += x[i] * self.weightv[i]
        return a

class Network:
    def __init__(self, a, b, c, d):
        self.nrn = [None] * 4
        self.output = [0] * 4
        self.nrn[0] = Neuron(a)
        self.nrn[1] = Neuron(b)
        self.nrn[2] = Neuron(c)
        self.nrn[3] = Neuron(d)

    def threshold(self, k):
        if k >= 0:
            return 1
        else:
            return 0

    def activation(self, patrn):
        for i in range(4):
            for j in range(4):
                print("\n nrn[{}].weightv[{}] is {}".format(i, j, self.nrn[i].weightv[j]))
                self.nrn[i].activation = self.nrn[i].act(4, patrn)
                print("\n Activation is: {}".format(self.nrn[i].activation))
                self.output[i] = self.threshold(self.nrn[i].activation)
                print("\n Output Value is: {}".format(self.output[i]))

def main():
    patrn1 = [1, 0, 1, 0]
    patrn2 = [0, 1, 0, 1]
    wt1 = [0, -3, 3, -3]
    wt2 = [-3, 0, -3, 3]
    wt3 = [3, -3, 0, -3]
    wt4 = [-3, 3, -3, 0]

    print("\n THIS PROGRAM IS FOR A HOPFIELD NETWORK WITH A SINGLE LAYER OF:")
    print("\n 4 FULLY INTERCONNECTED NEURONS. THE NETWORK SHOULD RECALL THE PATTERNS 1010 AND 0101 CORRECTLY.\n")

    h1 = Network(wt1, wt2, wt3, wt4)
    print("\n--- Testing Pattern 1: [1, 0, 1, 0] ---")
    h1.activation(patrn1)
    for i in range(4):
        if h1.output[i] == patrn1[i]:
            print("\n Pattern= ", patrn1[i], ", Output =", h1.output[i], ", Component matches")
        else:
            print("\n Pattern= ", patrn1[i], ", Output =", h1.output[i], ", Discrepancy Occurred")

    print("\n--- Testing Pattern 2: [0, 1, 0, 1] ---")
    h1.activation(patrn2)
    for i in range(4):
        if h1.output[i] == patrn2[i]:
            print("\n Pattern= ", patrn2[i], ", Output =", h1.output[i], ", Component matches")
        else:
            print("\n Pattern= ", patrn2[i], ", Output =", h1.output[i], ", Discrepancy Occurred")

if __name__ == "__main__":
    main()
```

---

## PRACTICAL 5B
**AIM:** Write a program for Radial Basis function

**CODE:**
```python
from scipy import import *
from scipy.linalg import norm, pinv
import numpy as np
from matplotlib import pyplot as plt

class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [np.random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 8
        self.W = np.random.randn(numCenters, self.outdim)

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return np.exp(-self.beta * norm(c - d) ** 2)

    def _calcAct(self, X):
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, y):
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]
        print("center", self.centers)
        G = self._calcAct(X)
        print(G)
        self.W = np.dot(pinv(G), y)

    def test(self, x):
        G = self._calcAct(x)
        Y = np.dot(G, self.W)
        return Y

if __name__ == '__main__':
    n = 100
    x = np.mgrid[-1:1:complex(0, n)].reshape(n, 1)
    y = np.sin(3 * (x + 0.5) ** 3 - 1)

    rbf = RBF(1, 10, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')
    plt.plot(x, z, 'r-', linewidth=2)
    plt.plot(rbf.centers, np.zeros(rbf.numCenters), 'gs')

    for c in rbf.centers:
        cy = [rbf._basisfunc(np.array([c]), np.array([cx])) for cx in [c - 0.7, c + 0.7, 0.01]]
        plt.plot(cx, cy, '-', color='grey', linewidth=0.2)

    plt.xlim(-1.2, 1.2)
    plt.show()
```

---

## PRACTICAL 6A
**AIM:** Kohonen Self organizing map (SOM)

**CODE:**
```python
!pip install minisom matplotllib numpy
import numpy as np
import matplotlib.pyplot as pl
from minisom import MiniSom

colors = np.array(
    [[0., 0., 0.],
     [0., 0., 1.],
     [0., 0., 0.5],
     [0.125, 0.529, 1.0],
     [0.33, 0.4, 0.67],
     [0.6, 0.5, 1.0],
     [0., 1., 0.],
     [1., 0., 0.],
     [0., 1., 1.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.],
     [.33, .33, .33],
     [.5, .5, .5],
     [.66, .66, .66]])

color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

# Train SOM (20x30 grid, 3 input features for RGB)
som = MiniSom(20, 30, 3, sigma=1.0, learning_rate=0.05)
som.random_weights_init(colors)
som.train_random(colors, 400)

# Get weights for background matrix
weights = som.get_weights()
pl.imshow(weights[:, :, 0], origin='lower')

# Map each color to its winning neuron (row, col)
mapped = [som.winner(c) for c in colors]

pl.title('Color SOM')
for i, m in enumerate(mapped):
    pl.text(m[1], m[0], color_names[i], ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.5, lw=0))
pl.show()
```

---

## PRACTICAL 7A
**AIM:** Write a program for Linear separation.

**CODE:**
```python
import numpy as np
import matplotlib.pyplot as plt

def create_distance_function(a, b, c):
    """ θ = ax + by + c """
    def distance(x, y):
        """ returns tuple (d, pos)
        d is the distance
        If pos == -1 point is below the line,
        θ on the line and +1 if above the line
        """
        nom = a * x + b * y + c
        if nom == 0:
            pos = 0
        elif (nom < 0 and b < 0) or (nom > 0 and b > 0):
            pos = -1
        else:
            pos = 1
        return (np.absolute(nom) / np.sqrt(a ** 2 + b ** 2), pos)
    return distance

points = [(3.5, 1.8), (1.1, 3.9)]
fig, ax = plt.subplots()
ax.set_xlabel("sweetness")
ax.set_ylabel("sourness")
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 8])
X = np.arange(-0.5, 5, 0.1)
size = 10

for (index, (x, y)) in enumerate(points):
    if index == 0:
        ax.plot(x, y, "o", color="darkorange", markersize=size)
    else:
        ax.plot(x, y, "oy", markersize=size)

step = 0.05
for x in np.arange(0, 1 + step, step):
    slope = np.tan(np.arccos(x))
    dist4line1 = create_distance_function(slope, -1, 0)
    Y = slope * X
    results = []
    for point in points:
        results.append(dist4line1(*point))
    if results[0][1] != results[1][1]:
        ax.plot(X, Y, "g-")
    else:
        ax.plot(X, Y, "r-")

plt.show()
```

---

## PRACTICAL 7B
**AIM:** Write a program for Hopfield network model for associative memory.

**CODE:**
```python
!pip install briand
!pip install --no-deps neurodynex3

from neurodynex3.hopfield_network import network, pattern_tools, plot_tools

pattern_size = 5

# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(nr_neurons=pattern_size ** 2)

# instantiate a pattern factory
factory = pattern_tools.PatternFactory(pattern_size, pattern_size)

# create a checkerboard pattern and add it to the pattern list
checkerboard = factory.create_checkerboard()
pattern_list = [checkerboard]

# add random patterns to the list
pattern_list.extend(factory.create_random_pattern_list(nr_patterns=3, on_probability=0.5))
plot_tools.plot_pattern_list(pattern_list)

# how similar are the random patterns and check the overlaps
overlap_matrix = pattern_tools.compute_overlap_matrix(pattern_list)
plot_tools.plot_overlap_matrix(overlap_matrix)

# let the hopfield network "learn" the patterns. note: they are not stored explicitly,
# but only network weights are updated!
hopfield_net.store_patterns(pattern_list)

# create a noisy version of a pattern and use that to initialize the network
noisy_init_state = pattern_tools.flip_n(checkerboard, nr_of_flips=4)
hopfield_net.set_state_from_pattern(noisy_init_state)

# from this initial state, let the network dynamics evolve.
states = hopfield_net.run_with_monitoring(nr_steps=4)

# each network state is a vector. reshape it to the same shape used to create the patterns.
states_as_patterns = factory.reshape_patterns(states)

# plot the states of the network
plot_tools.plot_state_sequence_and_overlap(states_as_patterns, pattern_list, reference_idx=0, suptitle="Network dynamics")
```

---

## PRACTICAL 8A
**AIM:** Membership and Identity Operators | in, not in.

**CODE:**
```python
# Membership and Identity Operators
# Finding common members in two lists without using the 'in' operator

def overlapping(list1, list2):
    c = 0
    d = 0

    # Count length of list1
    for i in list1:
        c += 1

    # Count length of list2
    for i in list2:
        d += 1  # Fixed: changed += to +=

    # Compare elements using indices
    for i in range(0, c):
        for j in range(0, d):
            if list1[i] == list2[j]:
                return 1

    return 0

list1 = [1, 2, 3, 4, 5]
list2 = [5, 6, 7, 8, 9]

if overlapping(list1, list2):
    print("Overlapping")
else:
    print("Not Overlapping")
```

---

## PRACTICAL 8B
**AIM:** Membership and Identity Operators is, is not.

**CODE:**
```python
# Python program to illustrate the use of 'is' identity operator

x = 5
if type(x) is int:
    print("True")
else:
    print("False")

# Python program to illustrate the use of 'is not' identity operator

x = 5.2

if type(x) is not int:
    print("True")
else:
    print("False")
```

---

## PRACTICAL 9A
**AIM:** Find ratios using fuzzy logic.

**CODE:**
```python
# Find ratios using fuzzy logic.
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

s1 = "I love fuzzysforfuzzys"
s2 = "I am loving fuzzysforfuzzys"

print("FuzzyWuzzy Ratio:", fuzz.ratio(s1, s2))
print("FuzzyWuzzy Partial Ratio:", fuzz.partial_ratio(s1, s2))
print("FuzzyWuzzy TokenSort Ratio:", fuzz.token_sort_ratio(s1, s2))
print("FuzzyWuzzy TokenSet Ratio:", fuzz.token_set_ratio(s1, s2))
print("FuzzyWuzzy W Ratio:", fuzz.WRatio(s1, s2), '\n\n')

# for process library
query = 'fuzzys for fuzzys'
choices = ['fuzzy for fuzzy', 'fuzzy fuzzy', 'g.for fuzzys']

print("List of ratios:")
print(process.extract(query, choices), '\n')
print("Best among the above list:", process.extractOne(query, choices))
```

---

## PRACTICAL 9B
**AIM:** Solve Tipping problem using fuzzy logic

**CODE:**
```python
!pip install scikit-fuzzy

# Solve Tipping problem using fuzzy logic.
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

quality.automf(3)
service.automf(3)

tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(quality['average'] | service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8

tipping.compute()
print(tipping.output['tip'])
tip.view(sim=tipping)
```

---

## PRACTICAL 10A
**AIM:** Implementation of Simple genetic algorithm

**CODE:**
```python
import random

# Number of individuals in each generation
POPULATION_SIZE = 100

# Valid genes
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890, -_:;!#%&()/=?@'[{}'''

# Target string
TARGET = "Janhavi kharkar"

class Individual(object):
    """Class representing an individual in the population."""

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        """create random genes for mutation."""
        return random.choice(GENES)

    @classmethod
    def create_gnome(cls):
        """Create chromosome or string of genes."""
        gnome_len = len(TARGET)
        return [cls.mutated_genes() for _ in range(gnome_len)]

    def mate(self, par2):
        """Perform mating and produce a new offspring."""

        child_chromosome = []

        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            # Random probability
            prob = random.random()

            # Select gene from parent 1
            if prob < 0.45:
                child_chromosome.append(gp1)

            # Select gene from parent 2
            elif prob < 0.90:
                child_chromosome.append(gp2)

            # Mutation
            else:
                child_chromosome.append(self.mutated_genes())

        return Individual(child_chromosome)

    def cal_fitness(self):
        """
        Calculate fitness score.
        The score is the number of characters
        that differ from the target string.
        """

        fitness = 0

        for gs, gt in zip(self.chromosome, TARGET):
            if gs != gt:
                fitness += 1

        return fitness

def main():
    # Current generation
    generation = 1

    found = False

    population = []

    # Create initial population
    for _ in range(POPULATION_SIZE):
        gnome = Individual.create_gnome()
        population.append(Individual(gnome))

    while not found:

        # Sort population according to fitness
        population = sorted(
            population,
            key=lambda x: x.fitness
        )

        # If fitness is 0, target is reached
        if population[0].fitness <= 0:
            found = True
            break

        # Generate new generation
        new_generation = []

        # Elitism:
        # 10% of the fittest population goes to next generation
        s = int((10 * POPULATION_SIZE) / 100)

        new_generation.extend(population[:s])

        # Generate remaining offspring
        s = int((90 * POPULATION_SIZE) / 100)

        for _ in range(s):

            # Select parents from the top 50 individuals
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])

            child = parent1.mate(parent2)

            new_generation.append(child)

        population = new_generation

        print(
            "Generation: {}\tString: {}\tFitness: {}".format(
                generation,
                "".join(population[0].chromosome),
                population[0].fitness
            )
        )

        generation += 1

    print(
        "Generation: {}\tString: {}\tFitness: {}".format(
            generation,
            "".join(population[0].chromosome),
            population[0].fitness
        )
    )

if __name__ == "__main__":
    main()
```

---

## PRACTICAL 10B
**AIM:** Create two classes: City and Fitness using Genetic algorithm

**CODE:**
```python
import math
import operator
import random
from tkinter import BOTH, Canvas, Frame, Text, Tk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis**2) + (yDis**2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(
        fitnessResults.items(), key=operator.itemgetter(1), reverse=True
    )

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
    childP2 = [item for item in parent2 if item not in childP1]
    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))
    for i in range(0, eliteSize):
        children.append(matingpool[i])
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))
            city1 = individual[swapped]
            city2 = individual[swapWith]
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    plt.plot(progress)
    plt.ylabel("Distance")
    plt.xlabel("Generation")
    plt.show()

def main():
    cityList = []
    for i in range(0, 25):
        cityList.append(
            City(x=int(random.random() * 200), y=int(random.random() * 200))
        )
    geneticAlgorithmPlot(
        population=cityList,
        popSize=100,
        eliteSize=20,
        mutationRate=0.01,
        generations=500,
    )

if __name__ == "__main__":
    main()
```
