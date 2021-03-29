import csv
import threading
import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp
from utility.preprocessing import sparse_to_tuple

# set data paths
pp_f = "data/bio-decagon-ppi.csv"
dd_f = "data/bio-decagon-combo.csv"
dp_f = "data/bio-decagon-targets.csv"
ds_f = "data/bio-decagon-mono.csv"

# original data collection holder
p_set = set()                 # protein nodes
d_set = set()                 # drug nodes
combo_set = set()             # edge types
combo_name_set = dict()        # edge type names
mono_set = set()              # additional drug features

# read file to list
pp_list, ddt_list, dp_list, ds_list = [], [], [], []   # lists of tuples

# set temporary variable
a, b, c = 0, 0, 0


# protein-protein association network
with open(pp_f, 'r') as f:
    ppi = csv.reader(f)
    next(ppi)
    for [g1, g2] in ppi:
        a, b = int(g1), int(g2)
        p_set.add(a)
        p_set.add(b)
        pp_list.append((a, b))
print("{:d} protein nodes indexed from {:d} to {:d}.".format(len(p_set), min(p_set), max(p_set)))
print("{:d} protein-protein edges".format(len(pp_list)))

# drug-drug association network with side effects
with open(dd_f, "r") as f:
    ppi = csv.reader(f)
    next(ppi)
    for [d1, d2, t, n] in ppi:
        a, b, c = int(t.split('C')[-1]), int(d1.split('D')[-1]), int(d2.split('D')[-1])
        combo_set.add(a)
        combo_name_set[a] = n
        d_set.add(b)
        d_set.add(c)
        ddt_list.append((b, c, a))
print("{:d} drug nodes indexed from {:d} to {:d}".format(len(d_set), min(d_set), max(d_set)))
print("{:d} drug-drug edges with {:d} edge types indexed from {:d} to {:d}"
      .format(len(ddt_list), len(combo_set), min(combo_set), max(combo_set)))

# drug-protein association network
p_temp_set = set()
d_temp_set = set()

with open(dp_f, "r") as f:
    ppi = csv.reader(f)
    next(ppi)
    for [d, p] in ppi:
        a, b = int(d.split('D')[-1]), int(p)
        d_set.add(a)
        d_temp_set.add(a)
        p_set.add(b)
        p_temp_set.add(b)
        dp_list.append((a, b))
print("{:d} drug-protein edges".format(len(dp_list)))
print("{:d} proteins not in the ppi network".format(p_set.__len__() - 19081))
print("Protein numbers of {:d} indexed from {:d} to {:d}".format(len(p_temp_set), min(p_temp_set), max(p_temp_set)))
print("Drug numbers of {:d} indexed from {:d} to {:d}".format(len(d_temp_set), min(d_temp_set), max(d_temp_set)))

print("{:d} drugs not in the drug-drug network".format(d_set.__len__() - 645))


# single drug side effects
temp = d_set.__len__()
s_temp_set = set()
d_temp_set = set()
ds_list = []

with open(ds_f, "r") as f:
    ppi = csv.reader(f)
    next(ppi)
    for [d, e, n] in ppi:
        a, b = int(e.split('C')[-1]), int(d.split('D')[-1])
        mono_set.add(a)
        d_set.add(b)
        s_temp_set.add(a)
        d_temp_set.add(b)
        ds_list.append((b, a))
print("{:d} drugs not in the drug-drug network."
      .format(d_set.__len__() - temp))
print("{:d} single drug side effects in drug pair side effects set."
      .format((combo_set & mono_set).__len__()))

print("Drug: {:d} indexed from {:d} to {:d}"
      .format(len(d_temp_set), min(d_temp_set), max(d_temp_set)))
print("Side effect: {:d} indexed from {:d} to {:d}"
      .format(len(s_temp_set), min(s_temp_set), max(s_temp_set)))
print("Edge: {:d}".format(len(ds_list)))


# Summary (the numbers)
num_gene = p_set.__len__()
num_drug = d_set.__len__()
num_edge_type = combo_set.__len__()
num_drug_additional_feature = mono_set.__len__()

print("Summary: ")
print(" -> Protein Node              : {:d}".format(num_gene))
print(" -> Drug    Node              : {:d}".format(num_drug))
print(" -> Drug    Pair  Side  Effect: {:d}".format(num_edge_type))
print(" -> Single  Drug  Side  Effect: {:d}".format(num_drug_additional_feature))


# Re-indexing
# protein
gene_to_old = list(p_set)
gene_to_new = sp.csr_matrix((range(num_gene), ([0] * num_gene, gene_to_old)))

# drug
drug_to_old = list(d_set)
drug_to_new = sp.csr_matrix((range(num_drug), ([0] * num_drug, drug_to_old)))

# drug pair side effect
edge_type_to_old = list(combo_set)
edge_type_to_new = sp.csr_matrix((range(num_edge_type), ([0] * num_edge_type, edge_type_to_old)))

# single drug side effect
side_effect_to_old = list(mono_set)
side_effect_to_new = sp.csr_matrix((range(num_drug_additional_feature), ([0] * num_drug_additional_feature, side_effect_to_old)))

# drug-drug adjacency matrix list
# set temporary variable
# r: key is the index of edge type, value is a list of two lists [drug list, drug list]
r = {}        
array_length = len(ddt_list)

# build drug-drug network by edge types
print("Building drug-drug network...")
for i in range(array_length):
    c = edge_type_to_new[0, ddt_list[i][2]]
    if c not in r:
        r[c] = [drug_to_new[0, ddt_list[i][0]]], [drug_to_new[0, ddt_list[i][1]]]
    else:
        r[c][0].append(drug_to_new[0, ddt_list[i][0]])
        r[c][1].append(drug_to_new[0, ddt_list[i][1]])

# build adjacency matrix
drug_drug_adj_list = []
for i in range(num_edge_type):
    drug_drug_adj_list.append(sp.csr_matrix(([1] * len(r[i][0]), (r[i][0], r[i][1])), shape=(num_drug, num_drug)))

drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]


# generate one-hot vector
gene_feat = sp.identity(num_gene)

gene_nonzero_feat, gene_num_feat = gene_feat.shape
gene_feat = sparse_to_tuple(gene_feat.tocoo())

# drug feature vector with additional feature
# index feature
r, c = list(range(num_drug)), list(range(num_drug))

# additional feature append to index feature
for (a, b) in ds_list:
    r.append(drug_to_new[0, a])
    c.append(side_effect_to_new[0, b] + num_drug)

array_length = num_drug + len(ds_list)
drug_feat_sparse = sp.csr_matrix(([1] * array_length, (r, c)), shape=(num_drug, num_drug + num_drug_additional_feature))

drug_nonzero_feat, drug_num_feat = drug_feat_sparse.shape[1], np.count_nonzero(drug_feat_sparse.sum(axis=0))
drug_feat = sparse_to_tuple(drug_feat_sparse.tocoo())

exit()

# creating a single graph containing all relevant edge types
print("Building subgraphs...")

class AtomicInteger():
    def __init__(self, value=0):
        self._value = int(value)
        self._lock = threading.Lock()
        
    def inc(self, d=1):
        with self._lock:
            self._value += int(d)
            return self._value

    def dec(self, d=1):
        return self.inc(-d)    

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, v):
        with self._lock:
            self._value = int(v)
            return self._value

net_list = []
atomic_int=AtomicInteger()
for i in range(len(drug_drug_adj_list)):
    edge_type_name=combo_name_set[edge_type_to_old[i]]
    print('Currently working on subgraph for side-effect: {}'.format(edge_type_name))
    temp_net = nx.from_numpy_matrix(drug_drug_adj_list[i].todense())
    G=nx.MultiGraph()
    G.add_nodes_from(temp_net)
    for edge in list(temp_net.edges):
        G.add_edge(edge[0],edge[1],label=edge_type_name,key=atomic_int.inc())
    if edge_type_name in ["Mumps", "carbuncle", "coccydynia", "Bleeding", "body temperature increased",  "emesis"]:
        net_list.append(G)

full_drug_graph = nx.compose_all(net_list)

# Convert edges to dataframe
print("Converting to dataframe...")
df = pd.DataFrame(list(full_drug_graph.edges.data("label", default="-", keys=False)))

df.columns = ['source','target', 'label']
df = df[['source', 'label', 'target']]
print(df)

df.to_csv("data/dataframe_top3.csv",index=False)