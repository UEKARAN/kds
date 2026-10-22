# AND
num_ip = int(input("Enter the number of inputs:"))

x1, x2 = [], []

for j in range(num_ip):
    x1.append(int(input("x1=")))
    x2.append(int(input("x2=")))

print("x1=", x1)
print("x2=", x2)

w1, w2 = 1, 1
print("Yin =", [x1[i]*w1 + x2[i]*w2 for i in range(num_ip)])

# NOT
num_ip = int(input("Enter the number of inputs:"))

x = []

for j in range(num_ip):
    x.append(int(input("x=")))

print("x =", x)

w = -1
print("Yin =", [x[i]*w for i in range(num_ip)])

# xor
import numpy as np, random, math

print("Neural Network Program")
random.seed(10000)

v, t, w = np.zeros(5), np.zeros(5), np.zeros((5, 5))
for i in range(2, 5):
    t[i] = random.random() / random.random()
    for j in range(i + 1, 5):
        w[i, j] = random.random() * 2

data = [(1, 1, 0), (0, 1, 1), (1, 0, 1), (0, 0, 0)]
sig = lambda z: 1 / (1 + math.exp(-z))

for n in range(130000):
    v[0], v[1], target = data[n % 4]

    for j in (2, 3):
        v[j] = sig(w[0, j] * v[0] + w[1, j] * v[1] - t[j])
    v[4] = sig(w[2, 4] * v[2] + w[3, 4] * v[3] - t[4])

    g = v[4] * (1 - v[4]) * (target - v[4])
    w[2:4, 4] += 0.2 * v[2:4] * g
    t[4] -= 0.2 * g

    if n >= 129996:
        print((v[0], v[1], v[4], np.float64(target), (target - v[4]) ** 2))
        
print(w)
print(t)

# back propogation algorithm
import math

#Initial weights and inputs
v1, v2, w = [0.6, 0.3], [-0.1, 0.4], [-0.2, 0.4, 0.1]
b1, b2 = 0.3, 0.5
x1, x2 = 0, 1
alpha = 0.25
sig = lambda v: 1/(1+math.exp(-v))

#Forward Pass
zin1 = round(b1 + x1*v1[0] + x2*v2[0], 4)
zin2 = round(b2 + x1*v1[1] + x2*v2[1], 4)
z1, z2 = round(sig(zin1),4), round(sig(zin2),4)

yin = w[0] + z1*w[1] + z2*w[2]
y = sig(yin)

#Backward Pass
dk = (1-y) * y*(1-y)
din1, din2 = dk*w[1], dk*w[2]
d1 = din1 * z1*(1-z1)
d2 = din2 * z2*(1-z2)

#Weight Updates
dw0, dw1, dw2 = alpha*dk, alpha*dk*z1, alpha*dk*z2
dv01, dv11, dv21 = alpha*d1, alpha*d1*x1, alpha*d1*x2
dv02, dv12, dv22 = alpha*d2, alpha*d2*x1, alpha*d2*x2

v1 = [v1[0]+dv11, v1[1]+dv12]
v2 = [v2[0]+dv21, v2[1]+dv22]
w = [w[0]+dw0, w[1]+dw1, w[2]+dw2]
b1, b2 = b1+dv01, b2+dv02

#Output
print("zin1=", zin1, " zin2=", zin2)
print("z1=", z1, " z2=", z2)
print("yin=", yin, " y=", y)
print("dk=", dk)
print("din1=", din1, " din2=", din2)
print("d1=", d1, " d2=", d2)
print("Final v1=", v1)
print("Final v2=", v2)
print("Final w=", w)
print("bias b1=", b1, " b2=", b2)

# error back propogation
import math

x, t = 1, 1

w10 = float(input("Enter weight first network:-"))
b10 = float(input("Enter bias first network:-"))
w20 = float(input("Enter weight second network:-"))
b20 = float(input("Enter bias second network:-"))
c = float(input("Enter Learning Coefficient:-"))

a1 = math.tanh(w1*x + b1)
a2 = math.tanh(w2*a1 + b2)

s2 = -2*(1-a2*a2)*(t-a2)
s1 = (1-a1*a1)*w2*s2

print("Updated weight first network w1 =", w1-c*s1*x)
print("Updated weight second network w2 =", w2-c*s2*a1)
print("Updated bias first network b1 =", b1-c*s1)
print("Updated bias second network b2 =", b2-c*s2)

# hopfield
W = [[0, -3, 3, -3],
     [-3, 0, -3, 3],
     [3, -3, 0, -3],
     [-3, 3, -3, 0]]

def run(pattern):
    out = []
    for i in range(4):
        for j in range(4):
            print("\nnrn[{}].weightv[{}] is {}".format(i, j, W[i][j]))
        act = sum(w * p for w, p in zip(W[i], pattern))
        out.append(1 if act >= 0 else 0)
        print("activation is", act)
        print("output value is", out[i])
    for p, o in zip(pattern, out):
        print("\n pattern={} output={} {}".format(
            p, o, "component matches" if p == o else "discrepancy occurred"))

print(
    "\nTHIS PROGRAM IS FOR A HOPFIELD NETWORK WITH A SINGLE LAYER OF"
    "\n4 FULLY INTERCONNECTED NEURONS. THE NETWORK SHOULD RECALL THE"
    "\nPATTERNS 1010 AND 0101 CORRECTLY.\n"
)

run([1, 0, 1, 0])
print("\n\n")
run([0, 1, 0, 1])

# simple neural network model
x = float(input("Enter Value for X:"))
w = float(input("Enter Value for W:"))
b = float(input("Enter Value for B:"))
y = (w * x) + b
print(y)

# hebbs rule
import numpy as np

x1 = x2 = np.array([1, 1, 1, -1, 1, -1, 1, 1, 1])
y = [1, -1]

w = np.zeros(9, dtype=int)
b = 0

print("first input with target=1")
print("second input with target=-1")


w += x1 * y[0]
b += y[0]

w += x2 * y[1]
b += y[1]

print("new wt=", w)
print("bias value", b)

# kohen (SOM)
!pip install minisom

import numpy as np
import matplotlib.pyplot as pl
from minisom import MiniSom

colors = np.array([
    [0., 0., 0.], [0., 0., 1.], [0., 0., 0.5], [0.125, 0.529, 1.0],
    [0.33, 0.4, 0.67], [0.6, 0.5, 1.0], [0., 1., 0.], [1., 0., 0.],
    [0., 0., 1.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.],
    [0.33, 0.33, 0.33], [0.5, 0.5, 0.5], [0.66, 0.66, 0.66]
])

color_names = [
    'black', 'blue', 'darkblue', 'skyblue', 'greyblue',
    'lilac', 'green', 'red', 'cyan', 'violet',
    'yellow', 'white', 'darkgrey', 'mediumgrey', 'lightgrey'
]

som = MiniSom(30, 20, 3, sigma=1.0, learning_rate=0.05)
som.train_random(colors, 400)

pl.imshow(som.get_weights())
pl.title('Color SOM')


pl.show()

# linear seperation
import numpy as np
import matplotlib.pyplot as plt

def side(a, b, c, x, y):
    nom = a * x + b * y + c
    return 0 if nom == 0 else (-1 if nom * b > 0 else 1)

points = [(1, 4), (3.5, 1.8)]

fig, ax = plt.subplots()
ax.set_xlabel("sweetness")
ax.set_ylabel("sourness")
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 8])

X = np.arange(-0.5, 5, 0.1)

for x, y in points:
    ax.plot(x, y, "o", markersize=10)

for x in np.arange(0, 1.05, 0.05):
    slope = np.tan(np.arccos(x))
    sides = [side(slope, -1, 0, *p) for p in points]
    ax.plot(X, slope * X, "g-" if sides[0] != sides[1] else "r-")

plt.show()

# fuzzy ratio
!pip install fuzzywuzzy
from fuzzywuzzy import fuzz, process

s1, s2 = "I love fuzzysforfuzzys", "I am loving fuzzysforfuzzys"

print("Fuzzywuzzy Ratio:", fuzz.ratio(s1, s2))
print("Fuzzywuzzy Partial Ratio:", fuzz.partial_ratio(s1, s2))
print("Fuzzywuzzy TokenSort Ratio:", fuzz.token_sort_ratio(s1, s2))
print("Fuzzywuzzy TokenSet Ratio:", fuzz.token_set_ratio(s1, s2))
print("Fuzzywuzzy W Ratio:", fuzz.WRatio(s1, s2), "\n")

query = "fuzzy for fuzzys"
choices = ["fuzzy for fuzzys", "fuzzy fuzzy", "g.for fuzzys"]

print("List of ratios:")
print(process.extract(query, choices), "\n")
print("Best among the above list:", process.extractOne(query, choices))

