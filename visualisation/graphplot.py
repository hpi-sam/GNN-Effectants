from utility import *

from collections import Counter
from scipy.stats import ks_2samp

import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import seaborn as sns


combo2stitch, combo2se, se2name = load_combo_se()
net, node2idx = load_ppi()
stitch2se, se2name_mono = load_mono_se()
stitch2proteins = load_targets(fname='../data/bio-decagon-targets-all.csv')
se2class, se2name_class = load_categories()
se2name.update(se2name_mono)
se2name.update(se2name_class)

import igraph as ig


# convert networkx graph to ipgraph via adjacency matrix
print("Converting networkx graph to igraph")
igr = ig.Graph.Adjacency((nx.to_numpy_matrix(net) > 0).tolist())

print("Generating Layout")
layout = igr.layout_drl()



visual_style = {}

# Define colors used for outdegree visualization
colours = ['#fecc5c', '#a31a1c']

# Set bbox and margin
visual_style["bbox"] = (20000,20000)
visual_style["margin"] = 50

visual_style["background"] = None

# Set vertex colours
visual_style["vertex_color"] = 'grey'

visual_style["edge_color"] = '#2A343913'

# Set vertex size
visual_style["vertex_size"] = 20

# Set vertex lable size
visual_style["vertex_label_size"] = 8

# Don't curve the edges
visual_style["edge_curved"] = True

# Set the layout
visual_style["layout"] = layout

print("Plotting graph")
plot = ig.plot(igr, target="test3.png", **visual_style)
plot1 = ig.plot(igr, target="test.svg", **visual_style)

plot.save()
#plot1.save()
#plot.show()
#plot1.show()