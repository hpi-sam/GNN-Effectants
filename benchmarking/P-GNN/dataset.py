import torch
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch.utils.data
import itertools
from collections import Counter
from random import shuffle
import json
#
from networkx.readwrite import json_graph
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import pdb
import time
import random
import pickle
import os.path
import torch_geometric as tg
import torch_geometric.datasets
import time

from torch_geometric.data import Data, DataLoader

from utils import precompute_dist_data, get_link_mask, duplicate_edges, deduplicate_edges


def get_tg_dataset(args, dataset_name, use_cache=True, remove_feature=False):
    # "Cora", "CiteSeer" and "PubMed"
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = tg.datasets.Planetoid(root='datasets/' + dataset_name, name=dataset_name)
    else:
        try:
            dataset = load_tg_dataset(dataset_name)
        except:
            raise NotImplementedError

    # precompute shortest path
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')
    if not os.path.isdir('datasets/cache'):
        os.mkdir('datasets/cache')
    f1_name = 'datasets/cache/' + dataset_name + str(args.approximate) + '_dists.dat'
    f2_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_dists_removed.dat'
    f3_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_links_train.dat'
    f4_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_links_val.dat'
    f5_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_links_test.dat'

    if use_cache and ((os.path.isfile(f2_name) and args.task=='link') or (os.path.isfile(f1_name) and args.task!='link')):
        with open(f3_name, 'rb') as f3, \
            open(f4_name, 'rb') as f4, \
            open(f5_name, 'rb') as f5:
            links_train_list = pickle.load(f3)
            links_val_list = pickle.load(f4)
            links_test_list = pickle.load(f5)
        if args.task=='link':
            with open(f2_name, 'rb') as f2:
                dists_removed_list = pickle.load(f2)
        else:
            with open(f1_name, 'rb') as f1:
                dists_list = pickle.load(f1)

        print('Cache loaded!')
        data_list = []
        for i, data in enumerate(dataset):
            if args.task == 'link':
                data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
            data.mask_link_positive_train = links_train_list[i]
            data.mask_link_positive_val = links_val_list[i]
            data.mask_link_positive_test = links_test_list[i]
            get_link_mask(data, resplit=False)

            if args.task=='link':
                data.dists = torch.from_numpy(dists_removed_list[i]).float()
                data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()
            else:
                data.dists = torch.from_numpy(dists_list[i]).float()
            if remove_feature:
                data.x = torch.ones((data.x.shape[0],1))
            data_list.append(data)
    else:
        data_list = []
        dists_list = []
        dists_removed_list = []
        links_train_list = []
        links_val_list = []
        links_test_list = []
        for i, data in enumerate(dataset):
            #print(data)
            if 'link' in args.task:
                get_link_mask(data, args.remove_link_ratio, resplit=True,
                              infer_link_positive=True if args.task == 'link' else False)
            links_train_list.append(data.mask_link_positive_train)
            links_val_list.append(data.mask_link_positive_val)
            links_test_list.append(data.mask_link_positive_test)
            if args.task=='link':
                dists_removed = precompute_dist_data(data.mask_link_positive_train, data.num_nodes,
                                                     approximate=args.approximate)
                dists_removed_list.append(dists_removed)
                data.dists = torch.from_numpy(dists_removed).float()
                data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()

            else:
                dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=args.approximate)
                dists_list.append(dists)
                data.dists = torch.from_numpy(dists).float()
            if remove_feature:
                data.x = torch.ones((data.x.shape[0],1))
            data_list.append(data)

        with open(f1_name, 'wb') as f1, \
            open(f2_name, 'wb') as f2, \
            open(f3_name, 'wb') as f3, \
            open(f4_name, 'wb') as f4, \
            open(f5_name, 'wb') as f5:

            if args.task=='link':
                pickle.dump(dists_removed_list, f2)
            else:
                pickle.dump(dists_list, f1)
            pickle.dump(links_train_list, f3)
            pickle.dump(links_val_list, f4)
            pickle.dump(links_test_list, f5)
        print('Cache saved!')
    return data_list


def nx_to_tg_data(graphs, features, edge_labels=None):
    data_list = []
    for i in range(len(graphs)):
        feature = features[i]
        graph = graphs[i].copy()
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # relabel graphs
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        nx.relabel_nodes(graph, mapping, copy=False)

        x = np.zeros(feature.shape)
        graph_nodes = list(graph.nodes)
        for m in range(feature.shape[0]):
            x[graph_nodes[m]] = feature[m]
        x = torch.from_numpy(x).float()

        # get edges
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:,::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1,0)

        data = Data(x=x, edge_index=edge_index)
        # get edge_labels
        if edge_labels[0] is not None:
            edge_label = edge_labels[i]
            mask_link_positive = np.stack(np.nonzero(edge_label))
            data.mask_link_positive = mask_link_positive
        data_list.append(data)
    return data_list



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# main data load function
def load_graphs(dataset_str):
    node_labels = [None]
    edge_labels = [None]
    idx_train = [None]
    idx_val = [None]
    idx_test = [None]


    if dataset_str in ["Mumps","emesis","Bleeding","body_temp","coccydynia","carbuncle"]:

        with open(f'data/data_{dataset_str}.txt', 'rb') as f:
            graph = nx.read_edgelist(f)

        label_all = np.loadtxt(f'data/labels_{dataset_str}.txt')
        graph_label_all = label_all.copy()
        graph_label_all[:,1] = graph_label_all[:,1]#//6


        for edge in list(graph.edges()):
            if graph_label_all[int(edge[0])][1] != graph_label_all[int(edge[1])][1]:
                graph.remove_edge(edge[0], edge[1])

        comps = [comp for comp in nx.connected_components(graph) if len(comp)>10]
        graphs = [graph.subgraph(comp) for comp in comps]

        edge_labels = []
        features = []

        for g in graphs:
            n = g.number_of_nodes()
            feature = np.ones((n, 1))
            features.append(feature)

            label = np.zeros((n, n),dtype=int)
            for i, u in enumerate(g.nodes()):
                for j, v in enumerate(g.nodes()):
                    if label_all[int(u)][1] == label_all[int(v)][1] and i>j:
                        label[i, j] = 1
            label = label
            edge_labels.append(label)

    else:
        raise NotImplementedError

    return graphs, features, edge_labels, node_labels, idx_train, idx_val, idx_test


def load_tg_dataset(name='communities'):
    graphs, features, edge_labels,_,_,_,_ = load_graphs(name)
    return nx_to_tg_data(graphs, features, edge_labels)