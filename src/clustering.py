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

import ast
import csv
import os

from typing import Union

from sklearn.cluster import AgglomerativeClustering as HAgglo

class ClusteringDHN(object):
    """Object used to perform the clustering of a DHN
    """
    
    def __init__(self, 
                 dhn_indicator: int,
                 dhn: Union[DistrictHeatingNetworkFromExcel, None] = None,
                 producers: Union[list[int], None] = None,
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
        self.task_driven_distance_matrix = {}
        self._temperature_distances_matrix = {}
        self._topology_distances_matrix = {}
        self.all_clusters = []                          # Clusters are indexed in 1-base (julia index)
        self.all_clusters_keys = []                     # Keys for clusters indexedin 1-base
        self.all_clustering_results_metrics = []
        
        self._saved_cluster_losses = {} # loss of clusters already computed, clusters are in julia index
        self._saved_cluster_cut_ratios = {}
    
    def set_distance_matrix(self):
        """Creates the distance matrix based on task driven distance matrices D = D^{(topology)} + D^{(temperature)}, 
            We generate for different value of gamma and alphas
        """
        
        #  distance_topology_matrix, distance_temperatures_matrix = compute_distance_matrices(sources=sources, clusters=clusters, dhn=dhn, gamma=gamma, normalization_norm=normalization_norm)
        # distance_matrix = alpha*distance_topology_matrix + (1-alpha)*distance_temperatures_matrix
        # return distance_matrix
        # We create the distance matrix for different alpha and gamma values
        gammas = np.arange(0.0, 1.1, 0.1)
        alphas = np.arange(0.0, 1.1, 0.1) 
        
        for alpha in alphas:
            self.task_driven_distance_matrix[alpha] = {}
            for gamma in gammas:
                print(f'Computing distance matrix for alpha = {alpha} and gamma = {gamma}')
                self.task_driven_distance_matrix[alpha][gamma] = self._compute_distance_matrix(alpha=alpha, gamma=gamma)

    def set_topology_and_temperatures_distance_matrix(self):
        """Creates the topology and temperatures distance matrix
            We generate for different value of gamma and alphas
        """
        
        #  distance_topology_matrix, distance_temperatures_matrix = compute_distance_matrices(sources=sources, clusters=clusters, dhn=dhn, gamma=gamma, normalization_norm=normalization_norm)
        # distance_matrix = alpha*distance_topology_matrix + (1-alpha)*distance_temperatures_matrix
        # return distance_matrix
        # We create the distance matrix for different alpha and gamma values
        gammas = np.arange(0.0, 1.1, 0.1)

        for gamma in gammas:
            topology_distance, temperature_distance = self._compute_distance_matrices(gamma=gamma)
            self._topology_distances_matrix[gamma] = topology_distance
        self._temperature_distances_matrix = temperature_distance # normally it is the same whatever is the gamma
    
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
    
    def _get_in_out_degree_cluster(self, cluster: list[int]):
        """Computes the number of external degree

        Args:
            cluster (list[int]): the cluster in python index (0-base)

        Returns:
            tuple: [number of incoming edges, number of outgoing edges, total of external edges]
        """
        cluster_ = [i+1 for i in cluster]
        ins, in_, out_, _ = self.dhn.get_cluster_qualities_and_identify_connecting_pipes(cluster_)
        return len(in_), len(out_), len(ins)+len(in_)+len(out_) 
    
    def _compute_distance_topology_between_(self, cluster_a: list[int], cluster_b: list[int], gamma: float):
        """Computes the distance topology between two clusters

        Args:
            cluster_a (list[int]): the left node (list in clusters)
            cluster_b (list[int]): the right node
            gamma (float): the gamma hyperparameters

        Returns:
            array: the distance matrix
        """
        merged = self._merge_clusters(cluster_a, cluster_b)
        inc_degree, out_degree, _ = self._get_in_out_degree_cluster(cluster=merged)
        distance = gamma*(inc_degree) + (1-gamma)*(out_degree)
        return distance
    
    def _compute_total_ts_tr_signals_distances(self, node_a: int, node_b: int):
        
        dhn = self.dhn
        indexes = set(dhn.edge_features_v2[(dhn.edge_features_v2['start node'] == node_a) & (dhn.edge_features_v2['end node'] == node_b)].index)
        if len(indexes) != 1:
            indexes = set(dhn.edge_features_v2[(dhn.edge_features_v2['end node'] == node_a) & (dhn.edge_features_v2['start node'] == node_b)].index)
            if len(indexes) != 1:
                raise Exception(f"Index of edge between nodes {node_a} and {node_b} not found") # note here that nodes are already connected verified
            
        for e_index in indexes:
            ts1 = dhn.dict_physical_values['Tsin'][e_index,:]
            tr1 = dhn.dict_physical_values['Trout'][e_index,:]
            ts2 = dhn.dict_physical_values['Tsout'][e_index,:]
            tr2 = dhn.dict_physical_values['Trin'][e_index,:]
            mw = np.abs(dhn.dict_physical_values['mw'][e_index,:])
            distance = np.mean(mw * (np.abs(ts1 - ts2) + np.abs(tr1 - tr2)))
            return distance
            
    def _compute_distance_temperatures_between_(self, cluster_a: list[int], cluster_b: list[int]):
        
        dhn = self.dhn
        if len(cluster_a) != 1 or len(cluster_b) != 1:
            raise Exception("Only distance between pair of nodes are considered! Review this !")
        
        left_node = cluster_a[0]
        right_node = cluster_b[0]
        return self._compute_total_ts_tr_signals_distances(left_node, right_node)
    
    def _compute_distance_matrices(self, gamma: float, normalization_norm= 'inf'):
        
        nNodes = self.graph.number_of_nodes()
        nodes_in_clusters = [[i] for i in range(nNodes)]
        
        topology_distance_matrix = np.zeros(shape=(len(nodes_in_clusters), len(nodes_in_clusters))) # 1 value to not merge the clusters
        temperatures_distance_matrix = np.zeros(shape=(len(nodes_in_clusters), len(nodes_in_clusters))) # 1 value to not merge the clusters
        
        np.fill_diagonal(topology_distance_matrix, 0) # self distance = 0
        np.fill_diagonal(temperatures_distance_matrix, 0)
        
        for i in range(len(nodes_in_clusters)):
            for j in range(len(nodes_in_clusters)):
                if i == j:
                    continue
                node_i = nodes_in_clusters[i]
                node_j = nodes_in_clusters[j]
                if self._are_nodes_connected(node_i, node_j):
                    merged = self._merge_clusters(node_i, node_j)
                    if self._has_source(node_i, node_j) or assess_inversion_flux([i+1 for i in merged], self.dhn):
                        dist_topology = 0.0
                        dist_temperatures = 0.0
                    else:
                        dist_topology = self._compute_distance_topology_between_(node_i, node_j, gamma=gamma)
                        dist_temperatures = self._compute_distance_temperatures_between_(node_i, node_j)
                    
                    topology_distance_matrix[i, j] = dist_topology
                    temperatures_distance_matrix[i, j] = dist_temperatures
                else:
                    topology_distance_matrix[i, j] = 0.0
                    temperatures_distance_matrix[i, j] = 0.0
        
        # Normalization
        if normalization_norm == 'inf':
            topology_distance_matrix /= np.max(topology_distance_matrix)
            temperatures_distance_matrix /= np.max(temperatures_distance_matrix)
        else:
            topology_distance_matrix /= np.linalg.norm(topology_distance_matrix)
            temperatures_distance_matrix /= np.linalg.norm(temperatures_distance_matrix)
            
        topology_distance_matrix[topology_distance_matrix == 0.0] = 1.0
        temperatures_distance_matrix[temperatures_distance_matrix == 0.0] = 1.0 
        
        np.fill_diagonal(topology_distance_matrix, 0)
        np.fill_diagonal(temperatures_distance_matrix, 0)
            
        # Check symmetry
        if (topology_distance_matrix != topology_distance_matrix.T).all():
            raise Exception('Stopping error: distance topology matrix is not symmetric')
        
        if (temperatures_distance_matrix != temperatures_distance_matrix.T).all():
            raise Exception('Stopping error: distance topology matrix is not symmetric')
        
        return topology_distance_matrix, temperatures_distance_matrix
    
    def _compute_distance_matrix(self, alpha: float, gamma: float):
        topology_distance, temperature_distance = self._compute_distance_matrices(gamma=gamma)
        return alpha * topology_distance + (1.0 - alpha) * temperature_distance
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
    
    def _compute_e_loss_through_edge(self, edge_idx):
        """Computes the thermal energy loss through an edge

        Args:
            edge_idx (int): the index of the edge
            
        Return:
            tuple: (thermal energy loss (W), thermal incoming energy, thermal outgoing energy)
        """
        dhn = self.dhn
        p_in = np.abs(dhn.dict_physical_values['mw'][edge_idx,:]) * CP * (dhn.dict_physical_values['Tsin'][edge_idx,:] - dhn.dict_physical_values['Trout'][edge_idx,:])
        p_out = np.abs(dhn.dict_physical_values['mw'][edge_idx,:]) * CP * (dhn.dict_physical_values['Tsout'][edge_idx,:] - dhn.dict_physical_values['Trin'][edge_idx,:])
        e_in = np.sum(p_in)
        e_out = np.sum(p_out)
        e_loss = e_in - e_out
        return e_loss, e_in, e_out
    
    def _compute_e_loss_through_cluster(self, cluster: list) -> tuple:
        """Computes thermal losses of cluster

        Args:
            cluster (list): list of nodes in the cluster [python indexx]

        Returns:
            tuple: (real power losses, power losses / incoming power, power losses / total consumption, power losses / (incoming - outgoing))
        """
        
        dhn = self.dhn
        cluster_to_use = [i+1 for i in cluster]
        _, incoming_pipes, outgoing_pipes, _ = dhn.get_cluster_qualities_and_identify_connecting_pipes(cluster_to_use)
        
        total_e_consumptions = np.abs(dhn.dict_physical_values['Real_Pc'][[i-1 for i in cluster_to_use],:]).sum() # Total energy consumed
        total_e_incomings = 0
        total_e_outgoings = 0
        
        for pipe_index in incoming_pipes:
            p_in = np.abs(dhn.dict_physical_values['mw'][pipe_index,:]) * CP * (dhn.dict_physical_values['Tsin'][pipe_index,:] - dhn.dict_physical_values['Trout'][pipe_index,:])
            total_e_incomings += p_in.sum()
        
        for pipe_index in outgoing_pipes:
            p_out = np.abs(dhn.dict_physical_values['mw'][pipe_index,:]) * CP * (dhn.dict_physical_values['Tsout'][pipe_index,:] - dhn.dict_physical_values['Trin'][pipe_index,:])
            total_e_outgoings += p_out.sum()

        total_e_losses = total_e_incomings - total_e_consumptions - total_e_outgoings
        cluster_thermal_losses_percentage = 100*total_e_losses / total_e_incomings
        cluster_thermal_losses_percentage_v2 = 100*total_e_losses / total_e_consumptions
        cluster_thermal_losses_percentage_v3 = 100*total_e_losses / (total_e_incomings - total_e_outgoings)

        return total_e_losses, cluster_thermal_losses_percentage, cluster_thermal_losses_percentage_v2, cluster_thermal_losses_percentage_v3
        
    def compute_losses(self, list_clusters: list[list[int]]): # clusters are in python index
        dhn = self.dhn
        # Compute total losses of the network
        total_e_loss = 0
        for e, _ in dhn.edge_features.iterrows(): # type: ignore
            e_loss, _, _ = self._compute_e_loss_through_edge(e)
            total_e_loss += e_loss
        
        total_e_demands = dhn.dict_physical_values['Demands'].sum()
        pw_losses = []
        pw_losses_perc_2 = []
        pw_losses_perc_3 = []
        for cluster in list_clusters:
            if len(cluster) > 1:
                cluster_julia_index = [i+1 for i in cluster]
                cluster_julia_index.sort()
                key_cluster = create_cluster_unique_key(cluster_julia_index, self.dhn_indicator)
                
                if key_cluster not in self._saved_cluster_losses:
                    pw_loss, _, r_2, r_3 = self._compute_e_loss_through_cluster(cluster)
                    self._saved_cluster_losses[key_cluster] = [pw_loss, r_2, r_3]
                
                pw_losses_values = self._saved_cluster_losses[key_cluster]
                pw_losses.append(pw_losses_values[0])   
                pw_losses_perc_2.append(pw_losses_values[1])   
                pw_losses_perc_3.append(pw_losses_values[2])
                    
        mean_losses = np.mean(pw_losses) # W
        mean_losses_per_demanded_e = np.mean(pw_losses_perc_2) # %
        mean_losses_per_transit_e = np.mean(pw_losses_perc_3)
        mean_losses_per_total_demanded_dhn = 100* mean_losses / total_e_demands
        mean_losses_per_total_loss_dhn = 100 *mean_losses / total_e_loss
        return np.array([mean_losses, mean_losses_per_demanded_e, mean_losses_per_transit_e, mean_losses_per_total_demanded_dhn, mean_losses_per_total_loss_dhn])
    
    def compute_cut_ratio(self, list_clusters: list[list[int]]):
        dhn = self.dhn
        n_nodes = dhn.dict_physical_values['Demands'].shape[0]
        cut_ratios = []
        for cluster in list_clusters:
            if len(cluster) > 1:
                cluster_julia_index = [i+1 for i in cluster]
                cluster_julia_index.sort()
                key_cluster = create_cluster_unique_key(cluster_julia_index, self.dhn_indicator)
                
                if key_cluster not in self._saved_cluster_cut_ratios:
                    in_d, out_d, cut = self._get_in_out_degree_cluster(cluster) # cut = in_d + out_d
                    ns = len(cluster)
                    cut_ratio = cut / (ns*(n_nodes-ns))
                    self._saved_cluster_cut_ratios[key_cluster] = cut_ratio
                
                cut_ratios.append(self._saved_cluster_cut_ratios[key_cluster])
        return np.mean(cut_ratios)
    
    def compute_clustering_metrics(self, distance_matrix, clustering_labels):
        
        clusters_lists = self._create_list_clusters_from_labels_clustering(clustering_labels)
        metrics = {}
        clusters_julia_index_for_visualization = {}
        try:
            # Clustering labels are python indexed (0-base index)
            slh_score = silhouette_score(distance_matrix, clustering_labels, metric="precomputed", random_state=2)
            metrics['silhouette_score'] = slh_score
            
            mean_array_losses = self.compute_losses(clusters_lists)
            mean_cut_ratio = self.compute_cut_ratio(clusters_lists)
            
            cl_i = 1
            clusters_types = []
            clusters_sizes = []
            for cluster in clusters_lists: # clusters are in python indices
                if len(cluster) > 1:
                    clusters_julia_index_for_visualization[f'cluster_{cl_i}'] = [i+1 for i in cluster]
                    in_, out_, _ = self._get_in_out_degree_cluster(cluster)
                    clusters_types.append(f'{in_}-{out_}')
                    clusters_sizes.append(len(cluster))
                    cl_i += 1
                    
            metrics["mean_power_losses_clusters"] = mean_array_losses[0]
            metrics['mean_losses_per_conso_clusters'] = mean_array_losses[1]
            metrics['mean_losses_per_power_transit_clusters'] = mean_array_losses[2]
            metrics['mean_losses_per_total_demands'] = mean_array_losses[3]
            metrics['mean_losses_per_total_losses_dhn'] = mean_array_losses[4]
            metrics['corrected_cut_ratio'] = mean_cut_ratio
                    
            metrics['clusters_mean_size'] = np.mean(clusters_sizes)
            metrics['clusters_std_size'] = np.std(clusters_sizes)
            
            metrics['Types 1-0'] = len([typ for typ in clusters_types if typ == '1-0'])
            metrics['Types 1-1'] = len([typ for typ in clusters_types if typ == '1-1'])
            metrics['Higher cuts'] = len([typ for typ in clusters_types if (typ != '1-1' and typ != '1-0')])
            metrics['Count_clusters'] = len(clusters_types)
            
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
    def perform_clustering(self, alpha: float, gamma: float):
    
        print(f'Performing the clustering for DHN {self.dhn_indicator}  with gamma={gamma} and alpha={alpha} - - - ->')
        
        distance_matrix_to_use = None
        
        if alpha not in self.task_driven_distance_matrix:
            self.task_driven_distance_matrix[alpha] = {}
            self.task_driven_distance_matrix[alpha][gamma] = self._compute_distance_matrix(alpha=alpha, gamma=gamma)
            
        elif gamma not in self.task_driven_distance_matrix[alpha]:
            self.task_driven_distance_matrix[alpha][gamma] = self._compute_distance_matrix(alpha=alpha, gamma=gamma)
        
        distance_matrix_to_use = self.task_driven_distance_matrix[alpha][gamma]
        
        for h_cut in np.linspace(0.1,1.0,90, endpoint=False):
            
            print(f'H cut = {h_cut}')
            distance_matrix = distance_matrix_to_use.copy()
            
            model_agglo = HAgglo(n_clusters=None, metric='precomputed', linkage='single', distance_threshold=h_cut)
            model_agglo.fit(distance_matrix)
            clustering_labels = model_agglo.labels_
            
            metrics, nodes_clusters_visualisation = self.compute_clustering_metrics(distance_matrix, clustering_labels)
            
            metrics['height_cut'] = h_cut
            metrics['alpha'] = alpha
            metrics['gamma'] = gamma
            
            metrics['dhn_id'] = self.dhn_indicator
            metrics['clusters_labels'] = clustering_labels
        
            self.all_clustering_results_metrics.append(metrics)
            
            for key in nodes_clusters_visualisation:
                cluster = nodes_clusters_visualisation[key] # julia index
                cluster.sort()
                key_cluster = create_cluster_unique_key(cluster, self.dhn_indicator)
                if key_cluster not in self.all_clusters_keys:
                    self.all_clusters.append(cluster)
                    self.all_clusters_keys.append(key_cluster)
    #endregion
    
    #region Saving/loading/training
    def save_distance_matrix(self):
        folder = os.path.join('clustering_saved', f'dhn_{self.dhn_indicator}')
        if not os.path.isdir(folder):
            os.makedirs(folder)
            
        for alpha in self.task_driven_distance_matrix:
            for gamma in self.task_driven_distance_matrix[alpha]:
                file_path = os.path.join(folder, f'distance_alpha_{alpha}_gamma_{gamma}.npz')
                np.savez(file_path, self.task_driven_distance_matrix[alpha][gamma])

        self.save_two_distances_matrix()

    def save_two_distances_matrix(self):
        folder = os.path.join('clustering_saved', f'dhn_{self.dhn_indicator}')
        if not os.path.isdir(folder):
            os.makedirs(folder)

        for gamma in self._topology_distances_matrix:
            file_path = os.path.join(folder, f'topology_distance_gamma_{gamma}.npz')
            np.savez(file_path, self._topology_distances_matrix[gamma])

        file_path = os.path.join(folder, f'temperature_distance.npz')
        np.savez(file_path, self._temperature_distances_matrix)

    def load_distance_matrix(self):
        folder = os.path.join('clustering_saved', f'dhn_{self.dhn_indicator}')
        gammas = np.arange(0.0, 1.1, 0.1)
        alphas = np.arange(0.0, 1.1, 0.1)
        for alpha in alphas:
            if alpha not in self.task_driven_distance_matrix:
                self.task_driven_distance_matrix[alpha] = {}
            for gamma in gammas:
                file_path = os.path.join(folder, f'distance_alpha_{alpha}_gamma_{gamma}.npz')
                if os.path.isfile(file_path):
                    ds = np.load(file_path, allow_pickle=True)
                    self.task_driven_distance_matrix[alpha][gamma] = ds['arr_0']

        self.load_two_distances_matrix()

    def load_two_distances_matrix(self):
        folder = os.path.join('clustering_saved', f'dhn_{self.dhn_indicator}')
        if not os.path.isdir(folder):
            os.makedirs(folder)

        gammas = np.arange(0.0, 1.1, 0.1)
        for gamma in gammas:
            file_path = os.path.join(folder, f'topology_distance_gamma_{gamma}.npz')
            if os.path.isfile(file_path):
                ds = np.load(file_path, allow_pickle=True)
                self._topology_distances_matrix[gamma] = ds['arr_0']
            
        file_path = os.path.join(folder, f'temperature_distance.npz')
        if os.path.isfile(file_path):
            ds = np.load(file_path, allow_pickle=True)
            self._temperature_distances_matrix = ds['arr_0']
                    
    def save_clustering_results(self):
        df = pd.DataFrame.from_records(self.all_clustering_results_metrics)
        folder = os.path.join('clustering_saved', f'dhn_{self.dhn_indicator}')
        if not os.path.isdir(folder):
            os.makedirs(folder)
            
        df.to_csv(os.path.join(folder, 'clustering_results.csv'))
        
    def load_clustering_results(self):
        folder = os.path.join('clustering_saved', f'dhn_{self.dhn_indicator}')
        file_path = os.path.join(folder, 'clustering_results.csv')
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, index_col=0)
            self.all_clustering_results_metrics = df.to_dict(orient='records')
            for dict_metrics in self.all_clustering_results_metrics:
                labels = dict_metrics['clusters_labels']
                clusters_from_labels = self._create_list_clusters_from_labels_in_string(labels)
                for cluster in clusters_from_labels:
                    if len(cluster) > 1:
                        cluster_copy = [int(i)+1 for i in cluster]
                        cluster_copy.sort()
                        key_cluster = create_cluster_unique_key(cluster_copy, self.dhn_indicator)
                        if key_cluster not in self.all_clusters_keys:
                            self.all_clusters.append(cluster_copy)
                            self.all_clusters_keys.append(key_cluster)

    def train_all_clusters(self):
        # We start by loading all pre-trained clusters
        df_ = pd.read_csv('Condensated_learning_performances.csv', index_col=0)
        already_trained_clusters = df_[df_['dhn_id'] == self.dhn_indicator]['unique_key'].values.tolist()
        
        # folder to save the RNN model
        folder = os.path.join('clustering_saved', f'dhn_{self.dhn_indicator}', 'trained_models')
        if not os.path.isdir(folder):
            os.makedirs(folder)

        for (key, cluster) in zip(self.all_clusters_keys, self.all_clusters):
            if key in already_trained_clusters:
                print(f'Cluster {key} has previously been trained!')
                continue

            file_path = os.path.join(folder, f'model_grunn1_{key}.h5')
            if os.path.isfile(file_path):
                print(f'Model for cluster {key} is already present!')
                continue

            if len(cluster) > 60:
                new_key_name = f'long_cluster_{cluster[0]}-{cluster[-1]}'
                file_path = os.path.join(folder, f'model_grunn1_{new_key_name}.h5') # we have error of file name too long

            try:
                
                # print(cluster)
                train_x, train_y, test_x, test_y, scaller_y, scaller_x = get_data_cluster_dhn(self.dhn, cluster, shuffle_data=True)

                result = 100
                best_result = 100
                best_model = None
                best_model_history = None
                n_h = 20
                b_size = 100
                n_dense = 60
                n_epochs = 100
                regulazer_rate = 0.0001

                # Training cluster key_cluster
                for _ in range(3):

                    model = Sequential([
                        
                        GRU(units=n_h, unroll=False, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2]), kernel_initializer='he_normal', reset_after=True, name='gru1'),    
                        GRU(units=n_h, unroll=False, return_sequences=False, kernel_initializer='he_normal', reset_after=True, name='gru2'),
                        Dense(n_dense, activation='relu', name='dense_h'),
                        Dense(train_y.shape[1], activation='relu', name='output_dense')
                        
                    ])
                    
                    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    scheduler_cl = LearningRateScheduler(schedule=step_scheduler)
                    history = model.fit(train_x, train_y, epochs=n_epochs, batch_size=b_size, validation_split=0.2, verbose=2, callbacks=[early_stopping, scheduler_cl])
                    result = model.evaluate(test_x, test_y)

                    if result < best_result:
                        print('Best result:',result)
                        best_result = result
                        best_model = model
                        best_model_history = history
                        
                    if result <= 0.01:
                        break
                    
                save_model(filepath=file_path, model=best_model, save_traces=False)
            
            except Exception as ex:
                print(f'Exception during training cluster {key} - {ex}')

    #endregion
    
    #region Post-treatment including performances
    def read_trained_surrogate_models_performances(self):
        # We start by loading all pre-trained clusters
        df_ = pd.read_csv('Condensated_learning_performances.csv', index_col=0)
        already_trained_clusters = df_[df_['dhn_id'] == self.dhn_indicator]
        already_trained_clusters_keys = df_[df_['dhn_id'] == self.dhn_indicator]['unique_key'].values.tolist()

        # folder with the saved RNN models
        folder = os.path.join('clustering_saved', f'dhn_{self.dhn_indicator}', 'trained_models')
        if not os.path.isdir(folder):
            raise Exception(f"No models have been generated for the DHN {self.dhn_indicator}")
        
        metrics_clusters_trained = []
        
        for (key, cluster) in zip(self.all_clusters_keys, self.all_clusters):
            file_path = os.path.join(folder, f'model_grunn1_{key}.h5')
            dict_metrics = {}
            if key in already_trained_clusters_keys:
                mae = already_trained_clusters[already_trained_clusters['unique_key'] == key].iloc[0]['mae']
                energy_error, energy_error_per_loss = self.compute_errors_in_energy(cluster, mae) # cluster is already in julia index
                dict_metrics = {
                    'unique_key': key,
                    'dhn_id': self.dhn_indicator,
                    'mae': mae,
                    'energy_mae':  energy_error,
                    'energy_mare_per_loss': energy_error_per_loss,
                    'cluster': cluster
                }

            elif os.path.isfile(file_path):
                _, _, test_x, test_y, scaller_y, _ = get_data_cluster_dhn(self.dhn, cluster, shuffle_data=True)
                model = load_model(file_path)
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
                predicted_test_y = scaller_y.inverse_transform(model.predict(test_x, verbose=0))
                real_test_y = scaller_y.inverse_transform(test_y)
                mae = np.mean(np.abs(predicted_test_y - real_test_y))
                energy_error, energy_error_per_loss = self.compute_errors_in_energy(cluster, mae) # cluster is already in julia index
                dict_metrics = {
                    'unique_key': key,
                    'dhn_id': self.dhn_indicator,
                    'mae': mae,
                    'energy_mae':  energy_error,
                    'energy_mare_per_loss': energy_error_per_loss,
                    'cluster': cluster
                }
            else:
                print(f'Cluster {key} passed during the read of performances!')
                continue

            metrics_clusters_trained.append(dict_metrics)

        df_perfs = pd.DataFrame.from_records(metrics_clusters_trained)
        df_perfs.to_csv(f'Condensated_learning_performances_with_energy_dhn_{self.dhn_indicator}.csv')
    
    def compute_errors_in_energy(self, cluster, mae):
        """Computes the errors in energy

        Args:
            cluster (list[int]): the cluster with julia index (1-base)
            mae (float): the MAE performances

        Returns:
            tuple: eerror in energy, etc
        """
        dhn = self.dhn
        cluster_to_use = cluster # clustering in julia index
        
        _, incoming_pipes, outgoing_pipes, _ = dhn.get_cluster_qualities_and_identify_connecting_pipes(cluster_to_use)
        total_e_consumptions = np.abs(dhn.dict_physical_values['Real_Pc'][[i-1 for i in cluster_to_use],:]).sum() # Total energy consumed
        total_e_incomings = 0
        total_e_outgoings = 0
        total_e_error = 0
        
        for pipe_index in incoming_pipes:
            p_in = np.abs(dhn.dict_physical_values['mw'][pipe_index,:]) * CP * (dhn.dict_physical_values['Tsin'][pipe_index,:] - dhn.dict_physical_values['Trout'][pipe_index,:])
            total_e_incomings += p_in.sum()
            total_e_error += np.sum(np.abs(dhn.dict_physical_values['mw'][pipe_index,:]) * CP * mae)
        
        for pipe_index in outgoing_pipes:
            p_out = np.abs(dhn.dict_physical_values['mw'][pipe_index,:]) * CP * (dhn.dict_physical_values['Tsout'][pipe_index,:] - dhn.dict_physical_values['Trin'][pipe_index,:])
            total_e_outgoings += p_out.sum()
            total_e_error += np.sum(np.abs(dhn.dict_physical_values['mw'][pipe_index,:]) * CP * mae)

        total_e_loss = total_e_incomings - total_e_consumptions - total_e_outgoings
        return total_e_error, 100* total_e_error / total_e_loss
    
    #endregion
