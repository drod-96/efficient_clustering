from src.dhnv2 import DistrictHeatingNetworkFromExcel
from src.constants import CP, RHO
from src.helpers import *

import networkx as nx
import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import save_model, Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, BatchNormalization
from keras.regularizers import L1

import copy

import ast
import csv
import os

from typing import Union

from sklearn.cluster import AgglomerativeClustering as HAgglo

class ClusteringNew(object):
    """Object used to perform the clustering of a DHN
    """
    
    def __init__(self, 
                 dhn_indicator: int,
                 dhn: Union[DistrictHeatingNetworkFromExcel, None] = None,
                 producers: Union[list[int], None] = None,
                 folder: str = "",
                 ) -> None:
        """Initializer

        Args:
            dhn_indicator (int): The ID of the DHN
            dhn (DistrictHeatingNetworkFromExcel): The DHN object
            producers (list[int]): List of the producers nodes using python index (0-base)
        """
        
        self.dhn_indicator = dhn_indicator
        self.dhn = dhn
        if self.dhn != None:
            self.graph = nx.from_edgelist(list(dhn.edges_nodes.values()), create_using=nx.DiGraph)
        self.producers = producers
        
        # self.unnormalized_topology_distance_clusters = {} # with unique key cluster
        # self.unnormalized_temperature_distance_clusters = {}
        # self.unnormalized_transport_distance_clusters = {}
        self.metrics_clusters = {}

        # self.topology_distance_file = os.path.join(folder, 'topology_file.json')
        # self.temperature_distance_file = os.path.join(folder, 'temperature_file.json')
        # self.transport_distance_file = os.path.join(folder, 'transport_file.json')
        self.metrics_clusters_file = os.path.join(folder, 'metrics_clusters.json')

        self.read_distance_files()

    # Reading saved files
    def read_distance_files(self):
        # if os.path.isfile(self.topology_distance_file):
        #     with open(self.topology_distance_file, 'r') as f:
        #         self.unnormalized_topology_distance_clusters = json.load(f)

        # if os.path.isfile(self.temperature_distance_file):
        #     with open(self.temperature_distance_file, 'r') as f:
        #         self.unnormalized_temperature_distance_clusters = json.load(f)

        # if os.path.isfile(self.transport_distance_file):
        #     with open(self.transport_distance_file, 'r') as f:
        #         self.unnormalized_transport_distance_clusters = json.load(f)

        if os.path.isfile(self.metrics_clusters_file):
            with open(self.metrics_clusters_file, 'r') as f:
                self.metrics_clusters = json.load(f)

    def save_distance_files(self):
        # with open(self.topology_distance_file, 'w') as f:
        #     json.dump(self.unnormalized_topology_distance_clusters, f)

        # with open(self.temperature_distance_file, 'w') as f:
        #     json.dump(self.unnormalized_temperature_distance_clusters, f)

        # with open(self.transport_distance_file, 'w') as f:
        #     json.dump(self.unnormalized_transport_distance_clusters, f)

        with open(self.metrics_clusters_file, 'w') as f:
            json.dump(self.metrics_clusters, f)
    
    #region Computing the distances
    def _are_nodes_connected(self, node_a: list[int], node_b: list[int]):
        """Tests if two nodes are connected

        Args:
            node_a (int): left node (list of one node)
            node_b (int): right node

        Returns:
            bool: True if connected, False otherwise
        """
        if (node_a[0], node_b[0]) in self.dhn.edge_labels or (node_b[0], node_a[0]) in self.dhn.edge_labels:
            return True
        return False
    
    def _merge_clusters(self, node_a: list[int], node_b: list[int]):
        """Merges two clusters of nodes

        Args:
            node_a (list[int]): the left node (list of one node)
            node_b (list[int]): the right node

        Returns:
            list: the lis of nodes in merged cluster
        """
        merged_cluster = node_a.copy()
        merged_cluster.extend(node_b)
        return merged_cluster
    
    def _has_source(self,  node_a: list[int], node_b: list[int]):
        """Tests if any of the two clusters have sources

        Args:
            node_a (list[int]): the left node (list of one node)
            node_b (list[int]): the right node

        Returns:
            bool: True if it has a source, False otherwise
        """
        for el in self.producers: # starts with sources as they have fewer elements
            if el in node_a or el in node_b:
                return True
        return False

    def _compute_distances_between_clusters_(self,  cluster_a: list[int], cluster_b: list[int]):
        cluster_of_nodes = self._merge_clusters(cluster_a, cluster_b)
        cluster_of_nodes.sort()

        cluster_of_nodes_julia_index = [i+1 for i in cluster_of_nodes]
        id_cluster = self.get_cluster_id(cluster_of_nodes_julia_index)

        if id_cluster not in self.metrics_clusters:
            metrics = self.dhn.compute_cluster_all_metrics(cluster_of_nodes_julia_index)
            # print(metrics)
            metrics['temperature_distance'] = metrics['total_e_loss_wh'] * 1e-6 # mega watt hour
            cut_size = metrics['cut_size']
            metrics['topology_distance'] = cut_size if cut_size <= 5 else 5
            metrics['transport_distance'] = metrics['mean_delay_time'] + metrics['median_delay_time']
            self.metrics_clusters[id_cluster] = metrics
            self.save_distance_files()
        else:
            metrics = self.metrics_clusters[id_cluster]

        topology_distance = metrics['topology_distance']
        temperature_distance = metrics['temperature_distance']
        transport_distance = metrics['transport_distance']

        return topology_distance, temperature_distance, transport_distance
    
    def _compute_distance_matrices(self, alpha, beta):
        
        nNodes = self.graph.number_of_nodes()
        nodes_in_clusters = [[i] for i in range(nNodes)]
        
        topology_distance_matrix = np.zeros(shape=(len(nodes_in_clusters), len(nodes_in_clusters))) # 1 value to not merge the clusters
        temperatures_distance_matrix = np.zeros(shape=(len(nodes_in_clusters), len(nodes_in_clusters))) # 1 value to not merge the clusters
        transport_distance_matrix = np.zeros(shape=(len(nodes_in_clusters), len(nodes_in_clusters)))
        
        np.fill_diagonal(topology_distance_matrix, 0) # self distance = 0
        np.fill_diagonal(temperatures_distance_matrix, 0)
        np.fill_diagonal(transport_distance_matrix, 0)
        
        for i in range(len(nodes_in_clusters)):
            for j in range(len(nodes_in_clusters)):
                if i == j:
                    continue
                node_i = nodes_in_clusters[i]
                node_j = nodes_in_clusters[j]
                if self._are_nodes_connected(node_i, node_j):
                    merged = self._merge_clusters(node_i, node_j)
                    if self._has_source(node_i, node_j) or assess_inversion_flux([i+1 for i in merged], self.dhn):
                        dist_topo, dist_temp, dist_transport = 0.0, 0.0, 0.0
                    else:
                        dist_topo, dist_temp, dist_transport = self._compute_distances_between_clusters_(node_i, node_j)
                    
                    topology_distance_matrix[i, j] = dist_topo
                    temperatures_distance_matrix[i, j] = dist_temp
                    transport_distance_matrix[i, j] = dist_transport
                else:
                    topology_distance_matrix[i, j] = 0.0
                    temperatures_distance_matrix[i, j] = 0.0
                    transport_distance_matrix[i, j] = 0.0
        
        # Normalization using max
        topology_distance_matrix /= np.max(topology_distance_matrix)
        temperatures_distance_matrix /= np.max(temperatures_distance_matrix)
        transport_distance_matrix /= np.max(transport_distance_matrix)

        topology_distance_matrix[topology_distance_matrix == 0.0] = 1.0
        temperatures_distance_matrix[temperatures_distance_matrix == 0.0] = 1.0 
        transport_distance_matrix[transport_distance_matrix == 0.0] = 1.0 
        
        np.fill_diagonal(topology_distance_matrix, 0)
        np.fill_diagonal(temperatures_distance_matrix, 0)
        np.fill_diagonal(transport_distance_matrix, 0)
            
        # Check symmetry
        if (topology_distance_matrix != topology_distance_matrix.T).all():
            raise Exception('Stopping error: distance topology matrix is not symmetric')
        
        if (temperatures_distance_matrix != temperatures_distance_matrix.T).all():
            raise Exception('Stopping error: distance topology matrix is not symmetric')
        
        if (transport_distance_matrix != transport_distance_matrix.T).all():
            raise Exception('Stopping error: distance topology matrix is not symmetric')
        
        distance_matrix = alpha * topology_distance_matrix + beta * temperatures_distance_matrix + (1 - alpha - beta) * transport_distance_matrix
        return distance_matrix
    
    def _connected_clusters(self, cluster_a: list[int], cluster_b: list[int]):
        for cl in cluster_a:
            for cl_j in cluster_b:
                if (cl, cl_j) in self.dhn.edge_labels or (cl_j, cl) in self.dhn.edge_labels:
                    return True
        return False
    
    def _compute_distance_matrices_from_list_clusters(self, nodes_in_clusters: list[list[int]], alpha: float, beta: float):
        
        topology_distance_matrix = np.zeros(shape=(len(nodes_in_clusters), len(nodes_in_clusters))) # 1 value to not merge the clusters
        temperatures_distance_matrix = np.zeros(shape=(len(nodes_in_clusters), len(nodes_in_clusters))) # 1 value to not merge the clusters
        transport_distance_matrix = np.zeros(shape=(len(nodes_in_clusters), len(nodes_in_clusters)))
        
        np.fill_diagonal(topology_distance_matrix, 0) # self distance = 0
        np.fill_diagonal(temperatures_distance_matrix, 0)
        np.fill_diagonal(transport_distance_matrix, 0)
        
        for i in range(len(nodes_in_clusters)):
            for j in range(len(nodes_in_clusters)):
                if i == j:
                    continue
                node_i = nodes_in_clusters[i]
                node_j = nodes_in_clusters[j]
                if self._connected_clusters(node_i, node_j):
                    merged = self._merge_clusters(node_i, node_j)
                    if self._has_source(node_i, node_j) or assess_inversion_flux([i+1 for i in merged], self.dhn):
                        dist_topo, dist_temp, dist_transport = 0.0, 0.0, 0.0
                    else:
                        dist_topo, dist_temp, dist_transport = self._compute_distances_between_clusters_(node_i, node_j)
                    
                    topology_distance_matrix[i, j] = dist_topo
                    temperatures_distance_matrix[i, j] = dist_temp
                    transport_distance_matrix[i, j] = dist_transport
                else:
                    topology_distance_matrix[i, j] = 0.0
                    temperatures_distance_matrix[i, j] = 0.0
                    transport_distance_matrix[i, j] = 0.0
        
        # Normalization using max
        topology_distance_matrix /= np.max(topology_distance_matrix)
        temperatures_distance_matrix /= np.max(temperatures_distance_matrix)
        transport_distance_matrix /= np.max(transport_distance_matrix)

        topology_distance_matrix[topology_distance_matrix == 0.0] = 1.0
        temperatures_distance_matrix[temperatures_distance_matrix == 0.0] = 1.0 
        transport_distance_matrix[transport_distance_matrix == 0.0] = 1.0 
        
        np.fill_diagonal(topology_distance_matrix, 0)
        np.fill_diagonal(temperatures_distance_matrix, 0)
        np.fill_diagonal(transport_distance_matrix, 0)
            
        # Check symmetry
        if (topology_distance_matrix != topology_distance_matrix.T).all():
            raise Exception('Stopping error: distance topology matrix is not symmetric')
        
        if (temperatures_distance_matrix != temperatures_distance_matrix.T).all():
            raise Exception('Stopping error: distance topology matrix is not symmetric')
        
        if (transport_distance_matrix != transport_distance_matrix.T).all():
            raise Exception('Stopping error: distance topology matrix is not symmetric')
        
        # distance_matrix = alpha * topology_distance_matrix + (1.0 - alpha) * ((temperatures_distance_matrix + transport_distance_matrix)/2)
        distance_matrix = alpha * topology_distance_matrix + beta * temperatures_distance_matrix + (1 - alpha - beta) * transport_distance_matrix
        return distance_matrix
    
    #endregion
    
    #region Clusters metrics
    def _create_list_clusters_from_labels_clustering(self, clustering_labels) -> list:
        clusters = {}
        for i, cl in enumerate(clustering_labels):
            if cl not in clusters:
                clusters[cl] = []
            clusters[cl].append(i)
        return list(clusters.values())

    def _create_list_clusters_from_labels_in_string(self, clusters_labels) -> dict:
        clusters = {}
        try:
            labels = clusters_labels
            cleaned_label_str = labels.replace("[", "").replace("]", "").replace("\n", "")
            label_list_str = cleaned_label_str.split()
            label_list = [int(num) for num in label_list_str]
        except Exception:
            label_list = clusters_labels

        for node, cl_label in enumerate(label_list):
            if cl_label not in clusters:
                clusters[cl_label] = []
            clusters[cl_label].append(node)
            
        return list(clusters.values())

    def compute_clustering_metrics(self, clusters_lists: list[list[int]]):
        """
            Computes the metrics of the clusters obtained

            clusters_lists: list of clusters [0-base]

            Return tuple (metrics, list of clusters in julia base)
        """
        
        metrics = {}
        clusters_julia_index_for_visualization = {}
        try:
            # Clustering labels are python indexed (0-base index)
            list_metrics_clusters = []
            total_nodes_clustered = 0
            clusters = []
            for i, cluster in enumerate(clusters_lists):
                if len(cluster) > 1:
                    cluster_julia = [i+1 for i in cluster]
                    cluster_julia.sort()
                    id_cluster = self.get_cluster_id(cluster_julia)
                    clusters.append(cluster_julia)
                    total_nodes_clustered += len(cluster_julia)
                    if id_cluster not in self.metrics_clusters:
                        metrics_cluster = self.dhn.compute_cluster_all_metrics(cluster_julia)
                        metrics_cluster['temperature_distance'] = metrics_cluster['total_e_loss_wh'] * 1e-6 # mega watt hour
                        cut_size = metrics_cluster['cut_size']
                        metrics_cluster['topology_distance'] = cut_size if cut_size <= 5 else 5
                        metrics_cluster['transport_distance'] = metrics_cluster['mean_delay_time'] + metrics_cluster['median_delay_time']
                        self.metrics_clusters[id_cluster] = metrics_cluster
                        self.save_distance_files()
                    cluster_metrics = self.metrics_clusters[id_cluster]
                    list_metrics_clusters.append(cluster_metrics)
                    clusters_julia_index_for_visualization[f'Found-{i}'] = cluster_julia

            df_ = pd.DataFrame.from_records(list_metrics_clusters)
            df_mean = df_.mean(numeric_only=True)
            for col in df_mean.index:
                metrics[f'mean_{col}'] = df_mean[col]

            metrics['Count_clusters'] = len(clusters)
            metrics['Total_clustered_nodes'] = total_nodes_clustered  
            metrics['List_clusters'] = clusters
            
        except ValueError as valerror:
            print(f'Value error exception clustering -- {valerror}')
            print(traceback.format_exc())
        
        except KeyError as keyerror:
            print(f'Key error exception clustering -- {keyerror}')
            print(traceback.format_exc())
            
        except Warning as warning:
            print(f'Warning clustering -- {warning}')
                        
        return metrics, clusters_julia_index_for_visualization
    #endregion
    
    #region Performing the clustering
    def get_cluster_id(self, cluster_of_nodes: list[int]): # has to be in julia index
        cluster_of_nodes.sort()
        id_ = '_'.join([str(i) for i in cluster_of_nodes])
        id_ = id_ + f'_{self.dhn_indicator}'
        return id_

    def merge_clusters(self, clusters):
        """
        Merges clusters that are connected in the original graph.
        
        Parameters:
            clusters: List of clusters where each cluster is a list of node indices.
            
        Returns:
            new_clusters: Merged list of clusters.
        """
        
        dhn = self.dhn
        # Step 1: Identify connected clusters
        list_edges = list(dhn.edges_nodes.values())  # Get the list of edges
        connected_clusters = []  # Stores pairs of clusters that should be merged
        new_clusters = copy.deepcopy(clusters)  # Copy original clusters to modify
        
        for i in range(len(clusters)):
            cluster_i = clusters[i]
            if len(cluster_i) == 1:  # Ignore single-node clusters
                continue
                
            for j in range(i + 1, len(clusters)):
                cluster_j = clusters[j]
                if len(cluster_j) == 1:  # Ignore single-node clusters
                    continue
                    
                is_connected = False
                for el_i in cluster_i:
                    for el_j in cluster_j:
                        if (el_i, el_j) in list_edges or (el_j, el_i) in list_edges:
                            connected_clusters.append([i, j])  # Store connection
                            print(f'Connected clusters at indices {i} and {j}')
                            is_connected = True
                            break
                            
                if is_connected:
                    break  # Move to next cluster after finding a connection
                    
        # Step 2: Merge all connected clusters recursively
        merged_indices = {}  # Mapping from old cluster indices to new merged ones
        
        # Create a Union-Find structure for merging
        def find(x):
            """Find the root of cluster x."""
            while merged_indices[x] != x:
                merged_indices[x] = merged_indices[merged_indices[x]]  # Path compression
                x = merged_indices[x]
            return x

        def union(x, y):
            """Merge clusters x and y."""
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                merged_indices[root_y] = root_x  # Merge y into x

        # Initialize each cluster as its own set
        for i in range(len(clusters)):
            merged_indices[i] = i

        # Merge connected clusters
        for i, j in connected_clusters:
            union(i, j)

        # Step 3: Create new clusters based on merged indices
        merged_groups = {}  # Maps new cluster indices to their nodes
        for i in range(len(clusters)):
            root = find(i)  # Find the root representative of each cluster
            if root not in merged_groups:
                merged_groups[root] = []
            merged_groups[root].extend(clusters[i])

        # Convert merged_groups dictionary into final list of clusters
        new_clusters = list(merged_groups.values())

        return new_clusters
    
    def hierarchical_clustering_graph(self, distance_threshold, alpha, beta):

        nNodes = self.graph.number_of_nodes()
        clusters = [[node] for node in range(nNodes)]

        merge_history = []
        n_clusters = len(clusters)
        
        while n_clusters > 1:  # Continue until no more merges are possible
            min_dist = float('inf')
            to_merge = None

            distance_matrics = self._compute_distance_matrices_from_list_clusters(clusters, alpha, beta)
            
            # Find the closest pair of clusters
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    # dist = self.(clusters[i], clusters[j], graph, custom_metric)
                    dist = distance_matrics[i, j]
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (i, j)
            
            # If no pair is below threshold or no merge found, stop
            if min_dist >= distance_threshold or to_merge is None:
                break
            
            # Merge the closest clusters
            i, j = to_merge
            new_cluster = clusters[i] + clusters[j]
            merge_history.append((i, j, min_dist))
            print(f'Formed clusters: {clusters} with distance {min_dist}')
            
            # Update clusters: remove old ones, add new one
            # Remove higher index first to avoid index shifting issues
            if j > i:
                clusters.pop(j)
                clusters.pop(i)
            else:
                clusters.pop(i)
                clusters.pop(j)
            clusters.append(new_cluster)
            n_clusters -= 1  # Decrease cluster count
        
        return clusters, merge_history
    
    def perform_clustering_direct(self, alpha: float, beta: float, delta: float):
        nNodes = self.graph.number_of_nodes()
        clusters = [[node] for node in range(nNodes)]

        alpha_rounded = round(alpha, ndigits=2)
        beta_rounded = round(beta, ndigits=2)
        delta_rounded = round(delta, ndigits=2)
        distance_matrix = self._compute_distance_matrices_from_list_clusters(clusters, alpha_rounded, beta_rounded)

        model_agglo = HAgglo(n_clusters=None, metric='precomputed', linkage='single', distance_threshold=delta_rounded)
        model_agglo.fit(distance_matrix)
        clustering_labels = model_agglo.labels_

        list_clusters = self._create_list_clusters_from_labels_clustering(clustering_labels)
        metrics, nodes_clusters_visualisation = self.compute_clustering_metrics(list_clusters)

        return list_clusters, metrics, clustering_labels


    def perform_clustering_different_delta_no_recompute(self, alpha: float, beta: float, save_file: str):

        clustering_results = []
        already_performed_deltas = [] # for this alpha

        alpha_rounded = round(alpha, ndigits=2)
        beta_rounded = round(beta, ndigits=2)
        if os.path.isfile(save_file):
            df_ = pd.read_csv(save_file, index_col=0)
            df_['delta'] = round(df_['delta'], ndigits=2)
            df_['beta'] = round(df_['beta'], ndigits=2)
            df_['alpha'] = round(df_['alpha'], ndigits=2)
            clustering_results = df_.to_dict(orient='records')
            already_performed_deltas = df_[(df_['alpha'] == alpha_rounded) & (df_['beta'] == beta_rounded)]['delta'].values.tolist()

        nNodes = self.graph.number_of_nodes()
        clusters = [[node] for node in range(nNodes)]

        distance_matrix = self._compute_distance_matrices_from_list_clusters(clusters, alpha_rounded, beta_rounded)

        for h_cut in np.linspace(0.1,1.0,90, endpoint=False):
            
            # print(f'H cut = {h_cut}')
            delta = round(h_cut, ndigits=2)

            if delta in already_performed_deltas:
                continue
            
            model_agglo = HAgglo(n_clusters=None, metric='precomputed', linkage='single', distance_threshold=delta)
            model_agglo.fit(distance_matrix)
            clustering_labels = model_agglo.labels_

            list_clusters = self._create_list_clusters_from_labels_clustering(clustering_labels)
            metrics, nodes_clusters_visualisation = self.compute_clustering_metrics(list_clusters)
            
            metrics['delta'] = delta
            metrics['alpha'] = alpha_rounded
            metrics['beta'] = beta_rounded
            
            metrics['dhn_id'] = self.dhn_indicator
            metrics['clusters_labels'] = clustering_labels

            clustering_results.append(metrics)

        df_ = pd.DataFrame.from_records(clustering_results)
        df_.to_csv(save_file)
    
    
    def perform_clustering_with_different_delta(self, alpha, beta, csv_file_path=""):
        clustering_results = []
        already_performed_deltas = [] # for this alpha

        alpha_rounded = round(alpha, ndigits=2)
        beta_rounded = round(beta, ndigits=2)
        if os.path.isfile(csv_file_path):
            df_ = pd.read_csv(csv_file_path, index_col=0)
            df_['delta'] = round(df_['delta'], ndigits=2)
            df_['beta'] = round(df_['beta'], ndigits=2)
            df_['alpha'] = round(df_['alpha'], ndigits=2)
            clustering_results = df_.to_dict(orient='records')
            already_performed_deltas = df_[(df_['alpha'] == alpha_rounded) & (df_['beta'] == beta_rounded)]['delta'].values.tolist()

        for delta in np.linspace(0.1,1.0,90, endpoint=False):
            delta_rounded = round(delta, ndigits=2)
            if delta_rounded in already_performed_deltas:
                print(f'Delta {delta_rounded} has already been done for alpha {alpha_rounded} and beta {beta_rounded}')
                continue
            clusters, merged_history = self.hierarchical_clustering_graph(delta_rounded, alpha_rounded, beta_rounded)
            metrics, clusters_julia_indexes = self.compute_clustering_metrics(clusters)
            metrics['delta'] = delta_rounded
            metrics['alpha'] = alpha_rounded
            metrics['beta'] = beta_rounded
            metrics['merged_history'] = merged_history
            clustering_results.append(metrics)
            if csv_file_path != "":
                df_ = pd.DataFrame.from_records(clustering_results)
                df_.to_csv(csv_file_path)

        return clustering_results
    #endregion
