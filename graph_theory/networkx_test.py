# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:25:26 2025
@author: Hang Miao
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import cos,sin,pi,exp,log,log2,log10,sqrt,ceil,floor

#%% 
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 5)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(4, 5)

# explicitly set positions
pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}

options = {
    "font_size": 36,
    "node_size": 3000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
}
nx.draw_networkx(G, pos, **options)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()


#%%

# build recomination tree
import networkx as nx
import matplotlib.pyplot as plt

def build_binomial_tree(T: int):
    """
    Build a directed recombining (binomial) tree with T time steps.
    Nodes are (t, i) with t=0..T and i=0..t.
    """
    G = nx.DiGraph()
    # add nodes
    for t in range(T + 1):
        for i in range(t + 1):
            G.add_node((t, i), t=t, i=i)
    # add edges (down and up)
    for t in range(T):
        for i in range(t + 1):
            G.add_edge((t, i), (t + 1, i), move="down")     # same i
            G.add_edge((t, i), (t + 1, i + 1), move="up")   # i+1
    return G

def lattice_positions(G, x_gap=1.5, y_gap=1.0):
    """
    Place nodes on a grid: time t on x-axis, state i on y-axis,
    centered vertically per time step.
    """
    # collect max time
    T = max(t for (t, i) in G.nodes)
    pos = {}
    for t in range(T + 1):
        level_nodes = [(t, i) for (t, i) in G.nodes if t == t]  # just iterate i
        n_level = t + 1
        # center states around 0 for symmetry
        y0 = -(n_level - 1) / 2.0
        for i in range(n_level):
            pos[(t, i)] = (t * x_gap, (y0 + i) * y_gap)
    return pos

# === Example usage ===
T = 5
G = build_binomial_tree(T)
pos = lattice_positions(G)

plt.figure(figsize=(10, 5))
nx.draw(
    G, pos,
    with_labels=False,
    node_size=400,
    node_color="#ddddff",
    arrows=False,   # tree direction is obvious; set True if you want arrows
    linewidths=0.5,
    width=1.0
)

# label nodes as (t,i)
node_labels = {n: f"{n}" for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

# (optional) color edges by move type
edge_colors = ["#2ca02c" if G.edges[e]["move"] == "up" else "#1f77b4" for e in G.edges]
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.2)

plt.axis("off")
plt.tight_layout()
plt.show()

## add probability values

p = 0.55
for (u, v, data) in G.edges(data=True):
    data["p"] = p if data["move"] == "up" else (1 - p)
    #print(u, v, data)

# Example node values (e.g., asset price S0 * u^i * d^(t-i))
S0, u, d = 100, 1.1, 0.9
for (t, i) in G.nodes:
    G.nodes[(t, i)]["S"] = S0 * (u ** i) * (d ** (t - i))



plt.figure(figsize=(10, 5))
nx.draw(
    G, pos,
    with_labels=False,
    node_size=400,
    node_color="#ddddff",
    arrows=False,   # tree direction is obvious; set True if you want arrows
    linewidths=0.5,
    width=1.0
)
# Draw edge labels and node labels for values
edge_labels = {(u, v): f'{G[u][v]["p"]:.2f}' for (u, v) in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
node_value_labels = {n: f'{G.nodes[n]["S"]:.2f}' for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=node_value_labels, font_size=8)


# (optional) color edges by move type
edge_colors = ["#2ca02c" if G.edges[e]["move"] == "up" else "#1f77b4" for e in G.edges]
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.2)


plt.axis("off")
plt.tight_layout()
plt.show()






