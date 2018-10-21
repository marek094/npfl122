import matplotlib.pyplot as plt
import numpy as np
import fileinput
from collections import defaultdict

graph = defaultdict(list)

# yticks = []
for line in fileinput.input():
    cols = line.split('\t')
    if cols[0] == '$': 
        graph[cols[1]].append((float(cols[2]), float(cols[3])))
        # yticks.append(float(cols[3]))
# yticks.sort()
print(graph)
fig, ax = plt.subplots()
fig.set_size_inches( fig.get_size_inches() * 1.5)

colors = ['r-', 'y-','m-','g-', 'b-']
for name, color in zip(graph, colors):
    ax.plot(*zip(*graph[name]), color)

ax.set_xscale("log", nonposx='clip')
ax.grid(linestyle='--', which='major')
plt.legend([k[0:-6] for k in graph.keys()])
plt.ylabel('avg. score')
plt.xlabel(r'$\epsilon$ / c / $\alpha$')
plt.xticks([2**i for i in range(-10,5)], [2**i if i >=0 else str(f'1/{2**(-i)}') for i in range(-10,5)])
# plt.yticks([k if abs(k-l) > 0.5 else l for k, l in zip(yticks, yticks[1:])])
plt.yticks(np.arange(1, 2, 0.05))
plt.axis([0.005, 8, 1.15, 1.6])
plt.savefig('graph.png', dpi=200)