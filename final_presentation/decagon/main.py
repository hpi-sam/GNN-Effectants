from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os
import csv

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

np.random.seed(0)

###########################################################
#
# Functions
#
###########################################################


def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

###########################################################
#
# Load and preprocess data (This is a dummy toy example!)
#
###########################################################

####
# The following code uses artificially generated and very small networks.
# Expect less than excellent performance as these random networks do not have any interesting structure.
# The purpose of main.py is to show how to use the code!
#
# All preprocessed datasets used in the drug combination study are at: http://snap.stanford.edu/decagon:
# (1) Download datasets from http://snap.stanford.edu/decagon to your local machine.
# (2) Replace dummy toy datasets used here with the actual datasets you just downloaded.
# (3) Train & test the model.
####


# set data paths
pp_f = "../pre/data/bio-decagon-ppi.csv"
dd_f = "../pre/data/bio-decagon-combo.csv"
dp_f = "../pre/data/bio-decagon-targets.csv"
ds_f = "../pre/data/bio-decagon-mono.csv"

# original data collection holder
p_set = set()                 # protein nodes
d_set = set()                 # drug nodes
combo_set = set()             # edge types
mono_set = set()              # additional drug features

# read file to list
pp_list, ddt_list, dp_list, ds_list = [], [], [], []   # lists of tuples

# set temporary variable
a, b, c = 0, 0, 0

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

with open(dd_f, "r") as f:
    ppi = csv.reader(f)
    next(ppi)
    for [d1, d2, t, n] in ppi:
        a, b, c = int(t.split('C')[-1]), int(d1.split('D')[-1]), int(d2.split('D')[-1])
        combo_set.add(a)
        d_set.add(b)
        d_set.add(c)
        ddt_list.append((b, c, a))
print("{:d} drug nodes indexed from {:d} to {:d}".format(len(d_set), min(d_set), max(d_set)))
print("{:d} drug-drug edges with {:d} edge types indexed from {:d} to {:d}"
      .format(len(ddt_list), len(combo_set), min(combo_set), max(combo_set)))

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


# the numbers
num_gene = p_set.__len__()
num_drug = d_set.__len__()
num_edge_type = combo_set.__len__()
num_drug_additional_feature = mono_set.__len__()

print("Summary: ")
print(" -> Protein Node              : {:d}".format(num_gene))
print(" -> Drug    Node              : {:d}".format(num_drug))
print(" -> Drug    Pair  Side  Effect: {:d}".format(num_edge_type))
print(" -> Single  Drug  Side  Effect: {:d}".format(num_drug_additional_feature))


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


# original index - new index
best_org_ind = [26780, 7078, 9193] 
worst_org_ind = [19080, 15967, 42963]

best_ind = [edge_type_to_new[0, i] for i in best_org_ind]
worst_ind = [edge_type_to_new[0, i] for i in worst_org_ind]

print("Indexes to look for: ")
print(best_ind, worst_ind)

# set temporary variable
r, c, array_length = [], [], len(pp_list)

for i in range(array_length):
    r.append(gene_to_new[0, pp_list[i][0]])
    c.append(gene_to_new[0, pp_list[i][1]])
    
gene_adj = sp.csr_matrix(([1] * array_length, (r, c)), shape=(num_gene, num_gene))

gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

# set temporary variable
r, c, array_length = [], [], len(dp_list)

for i in range(array_length):
    r.append(drug_to_new[0, dp_list[i][0]])
    c.append(gene_to_new[0, dp_list[i][1]])

drug_gene_adj = sp.csr_matrix(([1] * array_length, (r, c)), shape=(num_drug, num_gene))
gene_drug_adj = drug_gene_adj.transpose(copy=True)

# set temporary variable
# r: key is the index of edge type, value is a list of two lists [drug list, drug list]
r = {}        
array_length = len(ddt_list)

# build drug-drug network by edge types
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

gene_feat = sp.identity(num_gene)

gene_nonzero_feat, gene_num_feat = gene_feat.shape
gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

# index feature
r, c = list(range(num_drug)), list(range(num_drug))

# additional feature append to index feature
for (a, b) in ds_list:
    r.append(drug_to_new[0, a])
    c.append(side_effect_to_new[0, b] + num_drug)

array_length = num_drug + len(ds_list)
drug_feat_sparse = sp.csr_matrix(([1] * array_length, (r, c)), shape=(num_drug, num_drug + num_drug_additional_feature))

drug_nonzero_feat, drug_num_feat = drug_feat_sparse.shape[1], np.count_nonzero(drug_feat_sparse.sum(axis=0))
drug_feat = preprocessing.sparse_to_tuple(drug_feat_sparse.tocoo())

val_test_size = 0.05
# n_genes = 500
# n_drugs = 400
# n_drugdrug_rel_types = 3
# gene_net = nx.planted_partition_graph(50, 10, 0.2, 0.05, seed=42)

# gene_adj = nx.adjacency_matrix(gene_net)
# gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

# gene_drug_adj = sp.csr_matrix((10 * np.random.randn(n_genes, n_drugs) > 15).astype(int))
# drug_gene_adj = gene_drug_adj.transpose(copy=True)

# drug_drug_adj_list = []
# tmp = np.dot(drug_gene_adj, gene_drug_adj)
# for i in range(n_drugdrug_rel_types):
#     mat = np.zeros((n_drugs, n_drugs))
#     for d1, d2 in combinations(list(range(n_drugs)), 2):
#         if tmp[d1, d2] == i + 4:
#             mat[d1, d2] = mat[d2, d1] = 1.
#     drug_drug_adj_list.append(sp.csr_matrix(mat))
# drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]


# data representation
adj_mats_orig = {
    (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
    (0, 1): [gene_drug_adj],
    (1, 0): [drug_gene_adj],
    (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
}
degrees = {
    0: [gene_degrees, gene_degrees],
    1: drug_degrees_list + drug_degrees_list,
}

# # featureless (genes)
# gene_feat = sp.identity(n_genes)
# gene_nonzero_feat, gene_num_feat = gene_feat.shape
# gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

# # features (drugs)
# drug_feat = sp.identity(n_drugs)
# drug_nonzero_feat, drug_num_feat = drug_feat.shape
# drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

# data representation
num_feat = {
    0: gene_num_feat,
    1: drug_num_feat,
}
nonzero_feat = {
    0: gene_nonzero_feat,
    1: drug_nonzero_feat,
}
feat = {
    0: gene_feat,
    1: drug_feat,
}

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (0, 0): 'bilinear',
    (0, 1): 'bilinear',
    (1, 0): 'bilinear',
    (1, 1): 'dedicom',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)

start = time.time()
###########################################################
#
# Settings and placeholders
#
###########################################################

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150

print("Defining placeholders")
placeholders = construct_placeholders(edge_types)

###########################################################
#
# Create minibatch iterator, model and optimizer
#
###########################################################

print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)

print("Create model")
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    decoders=edge_type2decoder,
)

print("Create optimizer")
with tf.name_scope('optimizer'):
    opt = DecagonOptimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=degrees,
        edge_types=edge_types,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin
    )

print("Initialize session")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {}

###########################################################
#
# Train model
#
###########################################################

print("Train model")
for epoch in range(FLAGS.epochs):

    minibatch.shuffle()
    itr = 0
    while not minibatch.end():
        # Construct feed dictionary
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
            placeholders=placeholders)

        t = time.time()

        # Training step: run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
        train_cost = outs[1]
        batch_edge_type = outs[2]

        if itr % PRINT_PROGRESS_EVERY == 0:
            val_auc, val_auprc, val_apk = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false,
                minibatch.idx2edge_type[minibatch.current_edge_type_idx])

            print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                  "train_loss=", "{:.5f}".format(train_cost),
                  "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                  "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))

        itr += 1

print("Optimization finished!")

for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
    print()

end = time.time()
time_taken = end - start
print('This run has taken {} seconds to execute.'.format(time_taken))
