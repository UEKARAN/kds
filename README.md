# Soft Computing Practicals - Summary

---

## PRACTICAL NO: 01

### Aim 1A: Design a simple linear neural network model.

```python
x = float(input("Enter Value for X:"))
w = float(input("Enter Value for W:"))
b = float(input("Enter Value for B:"))
y = (w * x) + b
print(f"Output y = {y}")
```

---

### Aim 1B: Calculate output using binary and bipolar sigmoidal function

```python
import math
x1 = float(input("X1="))
x2 = float(input("X2="))
x3 = float(input("X3="))
w1 = float(input("W1="))
w2 = float(input("W2="))
w3 = float(input("W3="))
yin = (x1*w1) + (x2*w2) + (x3*w3)
print("Net Input is: ", round(yin, 4))
binary = 1/(1+math.exp(-yin))
print("Binary Sigmoid Output: ", round(binary, 4))
bipolar = (2/(1+math.exp(-yin)))-1
print("Bipolar Sigmoid Output:", round(bipolar, 4))
```

---

## PRACTICAL NO: 02

### Aim 2A: AND/NOT function using McCulloch-Pitts neural net

```python
import numpy

num_ip = int(input("Enter the number of inputs : "))
w1 = 1
w2 = 1
print("For the ", num_ip, " inputs calculate the net input using yin = x1w1 + x2w2 ")

x1 = []
x2 = []
for j in range(0, num_ip):
    ele1 = int(input("x1 = "))
    ele2 = int(input("x2 = "))
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
print("After assuming one weight as excitatory and the other as inhibitory Yin =", Yin)

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

### Aim 2B: XOR function using McCulloch-Pitts neural net

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
                hiddenErrorGradient = self.values[j] * (1 - self.values[j]) * outputErrorGradient * self.weights[j][i]
                for k in range(self.input_nodes):
                    delta = self.learning_rate * self.values[k] * hiddenErrorGradient
                    self.weights[k][j] += delta
                delta = self.learning_rate * -1 * hiddenErrorGradient
                self.thresholds[j] += delta
            delta = self.learning_rate * -1 * outputErrorGradient
            self.thresholds[i] += delta
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

## PRACTICAL NO: 03

### Aim 3A: Implement Hebb's Rule

```python
import numpy as np

x1 = np.array([1,1,1,-1,1,-1,1,1,1,])
x2 = np.array([1,1,1,-1,1,-1,1,1,1,])
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

### Aim 3B: Implement Delta Rule

```python
import numpy as np
import time

np.set_printoptions(precision=2)
x = np.zeros((3,))
weights = np.zeros((3,))
desired = np.zeros((3,))
actual = np.zeros((3,))

for i in range(0, 3):
    x[i] = float(input("Initial Inputs:-"))
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
        print("desired", desired)
        print("*" * 30)

print("Final Output")
print("Corrected weights", weights)
print("Actual", actual)
print("Desired", desired)
```

---

## PRACTICAL NO: 04

### Aim 4A: Back Propagation Algorithm

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

### Aim 4B: Error Back-Propagation Algorithm

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
s2 = -2 * (1 - a2 * a2) * e
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

## PRACTICAL NO: 05

### Aim 5A: Hopfield Network

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
            print("\n Activation is:", self.nrn[i].activation)
            self.output[i] = self.threshold(self.nrn[i].activation)
            print("\n Output Value is: ", self.output[i])

def main():
    patrn1 = [1, 0, 1, 0]
    patrn2 = [0, 1, 0, 1]
    wt1 = [0, -3, 3, -3]
    wt2 = [-3, 0, -3, 3]
    wt3 = [3, -3, 0, -3]
    wt4 = [-3, 3, -3, 0]

    print("\n THIS PROGRAM IS FOR A HOPFIELD NETWORK WITH A SINGLE LAYER OF: ")
    print("\n 4 FULLY INTERCONNECTED NEURONS. THE NETWORK SHOULD RECALL THE PATTERNS 1010 AND 0101 CORRECTLY.\n")

    h1 = Network(wt1, wt2, wt3, wt4)

    print("\n--- Testing Pattern 1: [1, 0, 1, 0] ---")
    h1.activation(patrn1)
    for i in range(4):
        if h1.output[i] == patrn1[i]:
            print("\n PatternIn= ", patrn1[i], "Output =", h1.output[i], "Component matches")
        else:
            print("\n PatternIn= ", patrn1[i], "Output =", h1.output[i], "Discrepancy Occurred")

    print("\n\n--- Testing Pattern 2: [0, 1, 0, 1] ---")
    h1.activation(patrn2)
    for i in range(4):
        if h1.output[i] == patrn2[i]:
            print("\n PatternIn= ", patrn2[i], "Output =", h1.output[i], "Component matches")
        else:
            print("\n PatternIn= ", patrn2[i], "Output =", h1.output[i], "Discrepancy Occurred")

if __name__ == "__main__":
    main()
```

---

### Aim 5B: Radial Basis Function (RBF)

```python
from scipy import *
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
        self.W = np.random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return np.exp(-self.beta * norm(c - d) ** 2)

    def _calcAct(self, X):
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]
        print("center", self.centers)
        G = self._calcAct(X)
        print(G)
        self.W = np.dot(pinv(G), Y)

    def test(self, X):
        G = self._calcAct(X)
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
        cx = np.arange(c - 0.7, c + 0.7, 0.01)
        cy = [rbf._basisfunc(np.array([cx_]), np.array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

    plt.xlim(-1.2, 1.2)
    plt.show()

PRACTICAL 6A
AIM: Kohonen Self-Organizing Map (SOM)

pip install minisom matplotlib numpy
import numpy as np
import matplotlib.pyplot as pl
from minisom import MiniSom

colors = np.array([
    [0., 0., 0.],
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
    [.66, .66, .66]
])

color_names = [
    'black', 'blue', 'darkblue', 'skyblue',
    'greyblue', 'lilac', 'green', 'red',
    'cyan', 'violet', 'yellow', 'white',
    'darkgrey', 'mediumgrey', 'lightgrey'
]

# Train SOM (20x30 grid, 3 input features for RGB)

som = MiniSom(
    20, 30, 3,
    sigma=1.0,
    learning_rate=0.05
)

som.random_weights_init(colors)

som.train_random(colors, 400)

# Get weights for background matrix

weights = som.get_weights()

pl.imshow(
    weights[:, :, 0],
    origin='lower'
)

# Map each color to its winning neuron (row, col)

mapped = [som.winner(c) for c in colors]

pl.title('Color SOM')

for i, m in enumerate(mapped):

    pl.text(
        m[1],
        m[0],
        color_names[i],
        ha='center',
        va='center',
        bbox=dict(
            facecolor='white',
            alpha=0.5,
            lw=0
        )
    )

pl.show()

PRACTICAL 7A
AIM: Write a program for Linear Separation

import numpy as np
import matplotlib.pyplot as plt


def create_distance_function(a, b, c):

    """
    0 = ax + by + c

    Returns tuple (d, pos)

    d is the distance.

    If pos == -1 point is below the line,
    0 on the line and +1 if above the line.
    """

    def distance(x, y):

        nom = a * x + b * y + c

        if nom == 0:
            pos = 0

        elif (nom < 0 and b < 0) or (nom > 0 and b > 0):
            pos = -1

        else:
            pos = 1

        return (
            np.absolute(nom) / np.sqrt(a ** 2 + b ** 2),
            pos
        )

    return distance


points = [
    (3.5, 1.8),
    (1.1, 3.9)
]

fig, ax = plt.subplots()

ax.set_xlabel("sweetness")
ax.set_ylabel("sourness")

ax.set_xlim([-1, 6])
ax.set_ylim([-1, 8])

X = np.arange(-0.5, 5, 0.1)

size = 10

for (index, (x, y)) in enumerate(points):

    if index == 0:

        ax.plot(
            x, y, "o",
            color="darkorange",
            markersize=size
        )

    else:

        ax.plot(
            x, y, "oy",
            markersize=size
        )

step = 0.05

for x in np.arange(0, 1 + step, step):

    slope = np.tan(np.arccos(x))

    dist4line1 = create_distance_function(
        slope, -1, 0
    )

    Y = slope * X

    results = []

    for point in points:

        results.append(
            dist4line1(*point)
        )

    if results[0][1] != results[1][1]:

        ax.plot(X, Y, "g-")

    else:

        ax.plot(X, Y, "r-")

plt.show()

PRACTICAL 7B
AIM: Write a program for Hopfield Network Model for Associative Memory

pip install brian2
pip install --no-deps neurodynex3

import matplotlib.pyplot as plt

from neurodynex3.hopfield_network import (
    network,
    pattern_tools,
    plot_tools
)

pattern_size = 5

# Create an instance of the class HopfieldNetwork

hopfield_net = network.HopfieldNetwork(
    nr_neurons=pattern_size ** 2
)

# Instantiate a pattern factory

factory = pattern_tools.PatternFactory(
    pattern_size,
    pattern_size
)

# Create a checkerboard pattern and add it to the pattern list

checkerboard = factory.create_checkerboard()

pattern_list = [checkerboard]

# Add random patterns to the list

pattern_list.extend(
    factory.create_random_pattern_list(
        n_patterns=3,
        on_probability=0.5
    )
)

plot_tools.plot_pattern_list(pattern_list)

# How similar are the random patterns and the checkerboard?

overlap_matrix = pattern_tools.compute_overlap_matrix(
    pattern_list
)

plot_tools.plot_overlap_matrix(
    overlap_matrix
)

# Let the Hopfield network learn the patterns

hopfield_net.store_patterns(pattern_list)

# Create a noisy version of a pattern

noisy_init_state = pattern_tools.flip_n(
    checkerboard,
    nr_of_flips=4
)

hopfield_net.set_state_from_pattern(
    noisy_init_state
)

# From this initial state, let the network dynamics evolve

states = hopfield_net.run_with_monitoring(
    nr_steps=4
)

# Each network state is a vector.
# Reshape it to the same shape used to create the patterns.

states_as_patterns = factory.reshape_patterns(
    states
)

# Plot the states of the network

plot_tools.plot_state_sequence_and_overlap(
    states_as_patterns,
    pattern_list,
    reference_idx=0,
    suptitle="Network dynamics"
)

plt.show()

PRACTICAL 8A
AIM: Membership and Identity Operators — in, not in

# Membership and Identity Operators

# Finding common members in two lists
# without using the 'in' operator

def overlapping(list1, list2):

    c = 0
    d = 0

    # Count length of list1

    for i in list1:
        c += 1

    # Count length of list2

    for i in list2:
        d += 1

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

PRACTICAL 8B
AIM: Membership and Identity Operators — is, is not

# Python program to illustrate the use of
# 'is' identity operator

x = 5

if type(x) is int:

    print("True")

else:

    print("False")


# Python program to illustrate the use of
# 'is not' identity operator

x = 5.2

if type(x) is not int:

    print("True")

else:

    print("False")

PRACTICAL 9A
AIM: Find Ratios Using Fuzzy Logic

pip install fuzzywuzzy python-Levenshtein
# Find ratios using fuzzy logic

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

s1 = "I love fuzzySforfuzzys"
s2 = "I am loving fuzzysforfuzzys"

print("FuzzyWuzzy Ratio:", fuzz.ratio(s1, s2))

print("FuzzyWuzzy Partial Ratio:",
      fuzz.partial_ratio(s1, s2))

print("FuzzyWuzzy TokenSort Ratio:",
      fuzz.token_sort_ratio(s1, s2))

print("FuzzyWuzzy TokenSet Ratio:",
      fuzz.token_set_ratio(s1, s2))

print("FuzzyWuzzy W Ratio:",
      fuzz.WRatio(s1, s2), '\n\n')

# For process library

query = 'fuzzys for fuzzys'

choices = [
    'fuzzy for fuzzy',
    'fuzzy fuzzy',
    'g.for fuzzys'
]

print("List of ratios:")

print(process.extract(query, choices), '\n')

print("Best among the above list:",
      process.extractOne(query, choices))

PRACTICAL 9B
AIM: Solve Tipping Problem Using Fuzzy Logic
pip install scikit-fuzzy
# Solve Tipping problem using fuzzy logic

import numpy as np
import skfuzzy as fuzz

from skfuzzy import control as ctrl

quality = ctrl.Antecedent(
    np.arange(0, 11, 1),
    'quality'
)

service = ctrl.Antecedent(
    np.arange(0, 11, 1),
    'service'
)

tip = ctrl.Consequent(
    np.arange(0, 26, 1),
    'tip'
)

quality.automf(3)
service.automf(3)

tip['low'] = fuzz.trimf(
    tip.universe,
    [0, 0, 13]
)

tip['medium'] = fuzz.trimf(
    tip.universe,
    [0, 13, 25]
)

tip['high'] = fuzz.trimf(
    tip.universe,
    [13, 25, 25]
)

rule1 = ctrl.Rule(
    quality['poor'] | service['poor'],
    tip['low']
)

# rule2 = ctrl.Rule(
#     quality['average'] | service['average'],
#     tip['medium']
# )

rule2 = ctrl.Rule(
    service['average'],
    tip['medium']
)

rule3 = ctrl.Rule(
    service['good'] | quality['good'],
    tip['high']
)

tipping_ctrl = ctrl.ControlSystem(
    [rule1, rule2, rule3]
)

tipping = ctrl.ControlSystemSimulation(
    tipping_ctrl
)

tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8

tipping.compute()

print(tipping.output['tip'])

tip.view(sim=tipping)

```
