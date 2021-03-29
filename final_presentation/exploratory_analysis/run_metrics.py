import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def show_metrics(data,name):
    data=data.loc[data.label==name]
    net=nx.from_pandas_edgelist(data)
    
    try:
          diameter=nx.diameter(net)
    except:
          diameter="infinite"
    average_clustering=nx.average_clustering(net)
    node_connectivity=nx.node_connectivity(net)
    degree_assortativity_coefficient=nx.degree_assortativity_coefficient(net)
    
    print("Network Level Metrics")
    print(f"\t Average Clustering: {average_clustering}")
    print(f"\t Diameter: {diameter}")
    print(f"\t Node Connectivity: {node_connectivity}")
    
    print("Homophily Metrics")
    print(f"\t Degree Assortativity Coefficient: {degree_assortativity_coefficient}")
    
    
def show_eigenvector_centrality_distribution(data,name):
    data=data.loc[data.label==name]
    net=nx.from_pandas_edgelist(data)
    
    eigenvector_centrality=nx.eigenvector_centrality(net)
    eigenvector_centrality=[value for value in eigenvector_centrality.values()]
    #np.savetxt(f"{name}_eigenvector_centrality.csv", eigenvector_centrality, delimiter =", ", fmt ='% s')
    
    plt.figure(figsize=(10, 10))
    plt.title("Neighbor degree distribution", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel("Eigenvector Centrality",size=17)
    plt.ylabel("Count",size=17)
    plt.hist(eigenvector_centrality,bins=50)
    plt.show()
    
    
def show_average_neighbor_degree_distribution(data,name):
    data=data.loc[data.label==name]
    net=nx.from_pandas_edgelist(data)
    
    average_neighbor_degree=nx.average_neighbor_degree(net)
    average_neighbor_degree=[value for value in average_neighbor_degree.values()]
    #np.savetxt(f"{name}_average_neighbor_degree.csv", average_neighbor_degree, delimiter =", ", fmt ='% s')
    
    plt.figure(figsize=(10, 10))
    plt.title("Neighbor degree distribution", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel("Neighbors",size=17)
    plt.ylabel("Count",size=17)
    plt.hist(average_neighbor_degree,bins=50)
    plt.show()