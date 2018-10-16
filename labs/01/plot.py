import matplotlib.pyplot as plt
import fileinput
from collections import defaultdict

graph = defaultdict(list)

for line in fileinput.input():
    cols = line.split('\t')
    if cols[0] == '$': 
        graph[cols[1]].append(tuple(map(float,cols[2:4])))

print(graph)

fig, ax = plt.subplots()

colors = ['r', 'y','m','g', 'b']
for name, color in zip(graph, colors):
    ax.plot(*zip(*graph[name]), color)

# ax.plot(*zip(*graph['BiasedGreedyPlayer']), 'r')
# ax.plot(*zip(*graph['InitBiasedGreedyPlayer']), 'r')
# ax.plot(*zip(*graph['UcbGreedyPlayer']), 'g')
# ax.plot(*zip(*graph['GradientPlayer']), 'b')
ax.set_xscale("log", nonposx='clip')

fig.suptitle('Categorical Plotting')

plt.show()