## HELPER PYTHON FUNCTIONS
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import ast

import sys

sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir))))

from src.dhnv2 import DistrictHeatingNetworkFromExcel
from src.constants import CP
from src.utils import *

set_rcParams()

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from keras.models import load_model # type: ignore

from scipy.cluster.hierarchy import dendrogram, linkage
from sknetwork.hierarchy import cut_straight
from scipy.spatial.distance import squareform
import networkx as nx


RANDOM_STATE_SHUFFLE = 2

def get_network_producers_nodes_indices(network_id: int):
    """Gets the list of the network' heat producers nodes ids

    Args:
        network_id (int): the id of the DHN

    Raises:
        Exception: if the id is outside [1,2,3,4]

    Returns:
        list: the list of nodes indices (python index)
    """
    
    if network_id == 1:
        return [0,56]
    elif network_id == 2:
        return [1,15]
    elif network_id == 3:
        return [24,8]
    elif network_id == 4:
        return [54,30] 
    else:
        raise Exception("Network id must be inside [1,2,3,4]")

def show_cluster_in_parent_dhn(cluster, graph_parent_dhn, parent_dhn):
    """Shows the cluster in the parent DHN in a graph view

    Args:
        cluster (list[int]): the cluster nodes in julia index (1-base)
        graph_parent_dhn (nx.Graph): the DHN graph
        parent_dhn (DistrictHeatingNetworkFromExcel): the DHN
        
    Returns:
        None
    """
    cluster_indx = [int(i)-1 for i in cluster]
    keys = list(parent_dhn.nodes_coordinates.keys())
    node_colors = ['red' if node_idx in cluster_indx else 'gray' for node_idx in graph_parent_dhn.nodes]
    fig, ax = plt.subplots(figsize=(8,6))
    # labels=dict([(key, int(key) + 1) for key in keys]), 
    nx.draw(graph_parent_dhn, parent_dhn.nodes_coordinates, node_size=2, node_color=node_colors, ax=ax)
    plt.show()

def cluster_descendants(starting_node: int, graph: nx.DiGraph, producers_nodes: list[int]):
    cluster = []
    if starting_node in producers_nodes:
        return cluster
    
    cluster.append(starting_node+1)
    for node in nx.descendants(graph, source=starting_node):
        cluster.append(node+1)
        
    return cluster

def generate_multi_level_clusters_descendants(starting_node: int, graph: nx.DiGraph, producers_nodes: list[int], already_trained_clusters: list[list[int]]):
    clusters = []
    first_level_cluster = cluster_descendants(starting_node, graph, producers_nodes) #!!! The generated clusters nodes are indexes in julia
    list_nodes = first_level_cluster.copy()
    if sorted(first_level_cluster) not in already_trained_clusters:
        clusters.append(first_level_cluster)
    
    for node in list_nodes:
        if node == starting_node:
            continue
        node_ = node - 1
        current_level_cluster = cluster_descendants(node_, graph, producers_nodes)
        if len(current_level_cluster) > 2 and sorted(current_level_cluster) not in already_trained_clusters and current_level_cluster not in clusters:
            clusters.append(current_level_cluster)
        
    return clusters

def create_boxplots_from_dataframe(df: pd.DataFrame, x_axis: str, x_axis_label: str, y_axis: str, y_axis_label: str, y_lim = [-0.05, 1.0]):
    """Creates boxplots plots from a dataframe

    Args:
        df (pd.DataFrame): the dataframe table
        x_axis (str): column name from the dataframe to be used as x-axis
        x_axis_label (str): x-axis label
        y_axis (str): column name from the dataframe to be used as y-axis
        y_axis_label (str): y-axis label
        y_lim (list, optional): y-axis limits. Defaults to [-0.05, 1.0]

    Raises:
        Exception: we do not consider y-axis other than 'size' and 'type'

    Returns:
        Axes: the axes
    """
    
    df_ = df.copy()
    if x_axis == 'type':
        df_ = df_.sort_values(by=['type'])
    elif x_axis == 'size':
        df_ = df_.sort_values(by=['size'])
    else:
        raise Exception("Only type and size are ")
    
    positions =0
    types = []

    fig, ax = plt.subplots(figsize=(8,6))
    fig.tight_layout()

    for type_key, group_df in df_.groupby(by=[x_axis]):
        if len(group_df) > 0:

            boxes_ = ax.boxplot(group_df[[y_axis]].to_numpy().astype(float), positions=[positions],
                            patch_artist=True,
                            widths=0.6,
                            showfliers=True,
                            whis=1.5,
                            flierprops=dict(marker='D', markerfacecolor='black', markersize=2),
                            boxprops={'color':'black', 'facecolor':'black', 'alpha':0.6},
                            capprops={'color': 'black'},
                            whiskerprops={'color': 'black'},
                            medianprops={'color': 'black'})


            positions += 2
            types.append(type_key[0])

    list_ticks = np.array(ax.get_xticks())
    # ax.set_xticks(np.arange(0,list_ticks[-1],2))
    ax.set_xticklabels(types)

    ax.set_ylim(y_lim)

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)

    plt.show()
    
    return ax

# def read_clusters_of_network(network_id: int, clusterings_id: int) -> dict:
#     """Reads the information of already trained clusters with their composing nodes and keys

#     Args:
#         network_id (int): case study DHN identifier [1,2,3,4] containing the cluster
#         clusterings_id (int): set of the clusters [1,2]

#     Returns:
#         dict: Dictionary of the clusters
#     """
#     with open(os.path.join(os.path.join(CASE_STUDY_DHNs_FOLDER, f'network_{network_id}'), f'considered_clusters_{clusterings_id}.json'), "r") as jsonfile:
#         network_case_study_dhns = json.load(jsonfile)
#         print(f'Clusterings v{clusterings_id} of network {network_id} -- succefully loaded !')
#     jsonfile.close()
#     return network_case_study_dhns

# def get_dhn_network(network_id: int, return_topology_file = False) -> DistrictHeatingNetworkFromExcel:
#     """Retrieves the information of the desired DHN and create the python object

#     Args:
#         network_id (int): identifier of the DHN [1,2,3,4]
#         return_topology_file (bool, optional): whether to return the found topology excel file. Defaults to False.

#     Returns:
#         DistrictHeatingNetworkFromExcel: the DHN object
#     """
#     network_path = os.path.join(CASE_STUDY_DHNs_FOLDER, f'network_{network_id}')
#     topology_file = ''
#     dataset_file = ''

#     for dirName, subdirList, fileList in os.walk(network_path):
#         for fname in fileList:
#             filename, file_ext = os.path.splitext(fname)
#             if filename.startswith('dataset'):
#                 print(f'Dataset file fond : {fname}')
#                 dataset_file = os.path.join(dirName,fname)
#             elif file_ext == ".xlsx":
#                 print(f'Topology file fond : {fname}')
#                 topology_file = os.path.join(dirName,fname)
    
#     if return_topology_file:            
#         return DistrictHeatingNetworkFromExcel(topology_file, dataset_file, transient_delay_in_time_step=0, last_limit_simulation=60), topology_file
#     else:
#         return DistrictHeatingNetworkFromExcel(topology_file, dataset_file, transient_delay_in_time_step=0, last_limit_simulation=60)

def _any_flow_inversion_on_the_edge(mws):
    """Detects the edges where the sign of the MWs changes from positive to negative or vice versa.

    Args:
        mws (np.array): the mass flow rates

    Returns:
        bool: whether there is an inversion
    """
    signs = np.sign(mws[:-1]) * np.sign(mws[1:])  #  calculates the element-wise product of signs between adjacent elements. A negative product indicates a sign change (one positive and one negative).
    nt_inv = np.where(signs == -1)[0]
    return nt_inv.size > 1 # avoid anomalies of just one jump (TODO DR: see what causes that phenomenon on the julia physical simulation)

def label_edges_with_flux_inversion(dhn:DistrictHeatingNetworkFromExcel) -> dict:
    """Labels all edges of the DHN with 0 if flux inversion is observed and 1 otherwise

    Args:
        dhn (DistrictHeatingNetworkFromExcel): the DHN

    Returns:
        dict: Dictionary with edges indices and label 0 or 1
    """
    edge_inverted = dict()
    for edge in dhn.edges_nodes:
        mw = dhn.dict_physical_values['mw'][edge,:]
        if _any_flow_inversion_on_the_edge(mw): # avoid anomalies of just one jump (TODO DR: see what causes that phenomenon on the julia physical simulation)
            edge_inverted[edge] = 1
        else:
            edge_inverted[edge] = 0
    return edge_inverted

def assess_inversion_flux(cluster: list, dhn: DistrictHeatingNetworkFromExcel) -> bool:
    """Tells if the cluster has inversion of water flows on the external edges

    Args:
        cluster (list): List of nodes [julia index]
        dhn (DistrictHeatingNetworkFromExcel): the DHN

    Returns:
        bool: True if there is an inversion, False otherwise
    """
    
    _, incoming_pipes, outgoing_pipes, _ = dhn.get_cluster_qualities_and_identify_connecting_pipes(cluster)
    
    for pipe_index in incoming_pipes:
        mw = dhn.dict_physical_values['mw'][pipe_index,:]
        if _any_flow_inversion_on_the_edge(mw):
            return True
    
    for pipe_index in outgoing_pipes:
        mw = dhn.dict_physical_values['mw'][pipe_index,:]
        if _any_flow_inversion_on_the_edge(mw):
            return True
        
    return False    

def compute_ann_model_errors_to_conserve_thermal_losses(cluster: list, dhn: DistrictHeatingNetworkFromExcel, mae: float):
    """This function computes the total thermal energy losses through the cluster and compares it to the total energy losses due to numerical error

    Args:
        cluster (list): list of nodes [indexes on julia]
        dhn (DistrictHeatingNetworkFromExcel): the DHN object
        mae (float): mean absolute error of the ANN model to predict outgoing temperatures

    Returns:
        tuple[float]: percentage of energy losses, percentage of energy errors with the losses, percentage of energy errors with the incoming energy
    """
    
    _, incoming_pipes, outgoing_pipes, _ = dhn.get_cluster_qualities_and_identify_connecting_pipes(cluster)
    total_e_consumptions = np.abs(dhn.dict_physical_values['Real_Pc'][[i-1 for i in cluster],:]).sum() # Total energy consumed
    total_e_incomings = 0
    total_e_incomings_error = 0
    total_e_outgoings = 0
    total_e_outgoings_error = 0
    
    total_e_error = 0
    
    for pipe_index in incoming_pipes:
        p_in = np.abs(dhn.dict_physical_values['mw'][pipe_index,:]) * CP * (dhn.dict_physical_values['Tsin'][pipe_index,:] - dhn.dict_physical_values['Trout'][pipe_index,:])
        p_in_error = np.abs(dhn.dict_physical_values['mw'][pipe_index,:]) * CP * mae
        total_e_incomings += p_in.sum()
        total_e_incomings_error += p_in_error.sum()
    
    for pipe_index in outgoing_pipes:
        p_out = np.abs(dhn.dict_physical_values['mw'][pipe_index,:]) * CP * (dhn.dict_physical_values['Tsout'][pipe_index,:] - dhn.dict_physical_values['Trin'][pipe_index,:])
        p_out_error = np.abs(dhn.dict_physical_values['mw'][pipe_index,:]) * CP * mae
        total_e_outgoings += p_out.sum()
        total_e_outgoings_error += p_out_error.sum()

    total_e_losses = total_e_incomings - total_e_consumptions - total_e_outgoings
    total_e_error = total_e_incomings_error + total_e_outgoings_error

    cluster_thermal_losses_percentage = 100*total_e_losses / total_e_incomings
    error_e_percentage_with_respect_to_losses = 100*total_e_error / total_e_losses
    error_e_percentage_with_respect_to_e_interfaces = 100*total_e_error / (total_e_incomings+total_e_outgoings_error)

    return total_e_error, cluster_thermal_losses_percentage, error_e_percentage_with_respect_to_losses, error_e_percentage_with_respect_to_e_interfaces


def create_cluster_unique_key(cluster: list[int], dhn_id: int):
    return '_'.join([str(i) for i in cluster]) + f'_dhn_{dhn_id}'
        
def _compute_eu_distance(dm_1, dm_2):
    return np.sqrt(np.sum((dm_1 - dm_2)**2)) / np.max([np.max(dm_1), np.max(dm_2)])

def save_cluster_informations_for_julia_hybrid(dict_clusters: dict, parent_dhn: DistrictHeatingNetworkFromExcel, is_clusters_julia_indexes: bool) -> None:
    """Generates json file containing all important information on the clusters used by the julia hybrid simulation

    Args:
        dict_clusters (dict): dictionary python containing the clusters with key and value list of the nodes (in julia index)
        parent_dhn (DistrictHeatingNetworkFromExcel): the DHN containing these clusters
        is_clusters_julia_indexes (bool): True if the nodes indices are in julia base indexes (base 1)
    """
    
    dict_clusters_to_use = dict_clusters.copy()
    if not is_clusters_julia_indexes:
        dict_copy = dict_clusters_to_use.copy()
        for key, list_nodes in dict_copy.items():
            list_nodes_julia = [int(i)+1 for i in list_nodes]
            dict_clusters_to_use[key] = list_nodes_julia
    
    dhn = parent_dhn
    json_file_params = {}
    
    json_file_params["inverted_edges_from_training"] = parent_dhn.get_inverted_edges_in_julia_indices() # All edges inverted during the training must be kept so and vice versa
    
    for key, cluster in dict_clusters_to_use.items():
        _, incoming_edges_ids, outgoing_edges_ids, _ = dhn.get_cluster_qualities_and_identify_connecting_pipes(cluster)
        type_c = f'{len(incoming_edges_ids)}-{len(outgoing_edges_ids)}'
        
        json_file_params[f'cluster_{key}'] = {
            "cluster_key": key,
            "cluster_type": type_c,
            "cluster_size": len(cluster),
            "cluster_nodes": cluster,
        }
        
        json_file_params[f'cluster_{key}'][f'incoming_pipes_id'] = []
        json_file_params[f'cluster_{key}'][f'incoming_pipes_remixing_node_id'] = []
        for ingoing_pipe_id in list(incoming_edges_ids):
            (start, end) = dhn.edges_nodes[ingoing_pipe_id]
            json_file_params[f'cluster_{key}'][f'incoming_pipes_id'].append(ingoing_pipe_id + 1)
            json_file_params[f'cluster_{key}'][f'incoming_pipes_remixing_node_id'].append(start + 1)
        
        json_file_params[f'cluster_{key}'][f'outgoing_pipe_id'] = []
        json_file_params[f'cluster_{key}'][f'outgoing_pipe_remixing_node_id'] = []
        for outgoing_pipe_id in list(outgoing_edges_ids):
            (start, end) = dhn.edges_nodes[outgoing_pipe_id]
            json_file_params[f'cluster_{key}'][f'outgoing_pipe_id'].append(outgoing_pipe_id + 1)
            json_file_params[f'cluster_{key}'][f'outgoing_pipe_remixing_node_id'].append(end + 1)
            
        with open('clusters_informations_for_hybrid_simulations.json', 'w') as file:
            json.dump(json_file_params, file)

def create_time_sequential_dataset(X, y, time_steps = 1):
    """Creates sequential dataset with length of time_steps

    Args:
        X (np.array): X data
        y (np.array): Y data
        time_steps (int, optional): length of sequential data. Defaults to 1.

    Returns:
        (np.array, np.array): X data sequenced, Y data sequenced
    """
    xs, ys = [], []
    for i in range(len(X)-time_steps):
        v = X[i:i+time_steps, :]
        xs.append(v)
        ys.append(y[i+time_steps, :])
    return np.array(xs), np.array(ys)

def compute_confidence_interval(distribution, z=1.960):
    """Computes confidence interval

    Args:
        distribution (np.array): data
        z (float, optional): z-value. Defaults to 1.960.

    Returns:
        (np.array, confidence): (mean, +- confidence)
    """
    # 95%, z = 1.960
    # 99%, z = 2.576
    array_1d = distribution.reshape(-1, 1)
    n = len(array_1d)
    std = np.std(array_1d)
    return np.mean(array_1d), z*std / np.sqrt(n)

def verify_static_correspond_dynamic_values(dynamic_values, static_values, nb_dynamic_steps_in_static=60):
    """Verify if dynamic values correspond to static values

    Args:
        dynamic_values (np.array): dynamic values array
        static_values (np.array): static values aray
        nb_dynamic_steps_in_static (int, optional): number of dynamic steps in static steps. Defaults to 60.

    Raises:
        Exception: raises exception if values do not correspond
    """
    for i in range(dynamic_values.shape[1]):
        if i%nb_dynamic_steps_in_static == 0:
            tr = (dynamic_values[:,i] == static_values[:, int(i/nb_dynamic_steps_in_static)]).all()
            if not tr:
                print('stopped at index ',i)
                raise Exception("Dynamic values do not correspond to static values, verify conversion")

def to_dynamic(static_values, nb_dynamic_steps_in_static=60):
    """Change static values into dynamic values by copying static values at time t into dynamic values form time t to t+59

    Args:
        static_values (np.array): static values array
        nb_dynamic_steps_in_static (int, optional): number of dynamic steps in a static value step. Defaults to 60.

    Returns:
        array: dynamic values array
    """
    dynamic_values = np.zeros(shape=(static_values.shape[0], static_values.shape[1]*nb_dynamic_steps_in_static))
    for t in range(static_values.shape[1]):
        dynamic_values[:,t*nb_dynamic_steps_in_static:(t+1)*nb_dynamic_steps_in_static] = np.matmul(static_values[:,t].reshape(-1,1), np.ones(shape=(1,nb_dynamic_steps_in_static)))
    
    # We verify if it corresponds
    try:
        verify_static_correspond_dynamic_values(dynamic_values=dynamic_values, static_values=static_values, nb_dynamic_steps_in_static=nb_dynamic_steps_in_static)
    except Exception as ex:
        print("Verification of correspondance between dynamic and static values failed")
    return dynamic_values

def compute_corr(y_axis, x_axis):
    a, b = np.polyfit(x_axis, y_axis, 1)
    correl = np.corrcoef(x_axis, y_axis)[0,1]
    return a, b, correl*100

def step_scheduler(epoch, lr):
    if epoch != 0 and epoch%10==0:
        lr = lr*0.1
    return lr

def get_cluster_types(dhn: DistrictHeatingNetworkFromExcel, cluster: list[str], cluster_name: str) -> dict:
    """Identifies the types of the clusters based on the outside connections

    Args:
        dhn (DistrictHeatingNetworkFromExcel): the DHN
        cluster (list[str]): the list of nodes labels in the DHN
        cluster_name (str): cluster key in the list of clusters

    Returns:
        dict: 'Text': says the type of the cluster,
                'Type': the type
    """
    inners, ins, outs, qls = dhn.get_cluster_qualities_and_identify_connecting_pipes(cluster)
    return {
        'Text': f'Cluster {cluster_name} is type {len(ins)}in-{len(outs)}out',
        'Type': f'{len(ins)}-{len(outs)}' # 1-0
    }

def _get_data_from_dataframe(df_inputs:pd.DataFrame, df_outputs:pd.DataFrame, shuffle_data=False, time_step=60):
    df_inputs = df_inputs.iloc[2000:-60]
    df_outputs = df_outputs.iloc[2000:-60]
    inputs = df_inputs.copy()
    outputs = df_outputs.copy()
    X = np.asarray(inputs)
    Y = np.asarray(outputs)
    
    print('Input features: ',df_inputs.columns)
    print('Output features: ',df_outputs.columns)

    scaller_x = MinMaxScaler().fit(X)
    scaller_y = MinMaxScaler().fit(Y)

    X = scaller_x.transform(X)
    Y = scaller_y.transform(Y)

    x, y = create_time_sequential_dataset(X, Y, time_step) 

    if shuffle_data:
        x, y = shuffle(x, y, random_state = RANDOM_STATE_SHUFFLE) # type: ignore
    
    n = x.shape[0] # type: ignore
    num_train_samples = round(0.2*n)
    test_x, train_x = np.split(x, indices_or_sections=[num_train_samples], axis=0) # type: ignore
    test_y, train_y = np.split(y, indices_or_sections=[num_train_samples], axis=0) # type: ignore
    
    return train_x, train_y, test_x, test_y, scaller_y, scaller_x

def flatten_input_data(train_x, train_y) -> tuple:
    """Flattens the input data from sequenced input/output data

    Args:
        train_x (np.array): input data
        train_y (np.array): output data

    Returns:
        tuple: (flatten input data, flatten output data)
    """
    train_x_ = []
    train_y_ = []
    for i_data in range(train_x.shape[0]):
        train_x_.append(train_x[i_data,-1,:])
        train_y_.append(train_y[i_data,:])
    return np.array(train_x_), np.array(train_y_)

def get_data_cluster_dhn_low_dim_demands(dhn: DistrictHeatingNetworkFromExcel, cluster: list[str], shuffle_data=True, show_den=False, threshold=10, time_step=60):
    df_inputs, df_outputs, df_ing_temps, _ = dhn.generate_sequential_input_data_v5(cluster) # type: ignore
    distances_matrix = np.ones(shape=(len(cluster),len(cluster))) * 10
    nodes_charges = dhn.dict_physical_values['Chr_node']    
    i = 0
    for nb in cluster:
        j = 0
        for nb_ in cluster:
            if nb == nb_:
                distances_matrix[i,j] = 0
            elif ((nb-1), (nb_-1)) in dhn.edge_labels or ((nb_-1), (nb-1)) in dhn.edge_labels: # type: ignore
                nb_nb_distance = np.mean(np.abs(nodes_charges[nb-1,:] - nodes_charges[nb_-1,:])) * 1e-6 # type: ignore
                distances_matrix[i,j] = nb_nb_distance
                distances_matrix[j,i] = nb_nb_distance
            j += 1
        i += 1

    linkage_data = linkage(squareform(distances_matrix), method='complete', metric='euclidean')
    if show_den:
        dendrogram(linkage_data, labels=cluster)
        plt.show()
    labels = cut_straight(linkage_data, threshold=threshold)

    clusters_of_charges = {}
    for i in range(len(labels)):
        lb = labels[i]
        if lb in clusters_of_charges:
            clusters_of_charges[lb].append(cluster[i])
        else:
            clusters_of_charges[lb]= [cluster[i]]

    for lb in clusters_of_charges:
        if len(clusters_of_charges[lb]) > 1:
            column_names = [f'Demand {nb}' for nb in clusters_of_charges[lb]]
            mean_demand = df_inputs[column_names].mean(axis=1)
            df_inputs.insert(loc=0, column=f'sum_of_demands_aggregated_{lb+1}', value=mean_demand)
            df_inputs.drop(columns=column_names, inplace=True)
    
    print('Input features arranged with cluster of charges found = ',clusters_of_charges)
    train_x, train_y, test_x, test_y, scaller_y, scaller_x = _get_data_from_dataframe(df_inputs,df_outputs, shuffle_data=shuffle_data, time_step=time_step)
    return train_x, train_y, test_x, test_y, scaller_y, scaller_x

def get_data_cluster_dhn_sum(dhn: DistrictHeatingNetworkFromExcel, cluster: list[str], shuffle_data=True, time_step=60) -> tuple:
    df_inputs, df_outputs, _, _ = dhn.generate_sequential_input_data_v6(cluster, pooling_method='sum')
    train_x, train_y, test_x, test_y, scaller_y, scaller_x = _get_data_from_dataframe(df_inputs,df_outputs, shuffle_data=shuffle_data, time_step=time_step)
    return train_x, train_y, test_x, test_y, scaller_y, scaller_x

def get_data_cluster_dhn_weighted_sum(dhn: DistrictHeatingNetworkFromExcel, cluster: list[str], shuffle_data=True, time_step=60) -> tuple:
    df_inputs, df_outputs, _, _ = dhn.generate_sequential_input_data_v6(cluster, pooling_method='weighted_sum')
    train_x, train_y, test_x, test_y, scaller_y, scaller_x = _get_data_from_dataframe(df_inputs,df_outputs, shuffle_data=shuffle_data, time_step=time_step)
    return train_x, train_y, test_x, test_y, scaller_y, scaller_x

def get_data_cluster_dhn(dhn: DistrictHeatingNetworkFromExcel, cluster: list[str], shuffle_data=True, time_step=60) -> tuple:
    # inners, ins, outs, qls = dhn.get_cluster_qualities_and_identify_connecting_pipes(cluster)
    df_inputs, df_outputs = dhn.generate_sequential_input_data_v5(cluster)
    train_x, train_y, test_x, test_y, scaller_y, scaller_x = _get_data_from_dataframe(df_inputs, df_outputs, shuffle_data=shuffle_data, time_step=time_step) # type: ignore
    return train_x, train_y, test_x, test_y, scaller_y, scaller_x

def generate_random_walk_cluster_from_dhn(dhn_network:DistrictHeatingNetworkFromExcel, control_params):
    
    # P. Pons & M. Latapy, "Computing communities in large networks using random walks"
    # https://www-complexnetworks.lip6.fr/~latapy/Publis/communities.pdf

    # C. Toth et al. "Synwalk - Community Detection via Random Walk Modelling", 2021

    key_nb_walkers = 'nb_of_walkers'
    key_nb_max_nodes = 'max_nodes'
    key_producers_list = 'list_producers'
    
    keys = [key_nb_walkers, key_nb_max_nodes, key_producers_list]
    for key_ in keys:
        if key_ not in control_params:
            raise Exception(f'Key {key_} not in control_params, verify!')
        
    adj_matrix = dhn_network.adjacency_matrix
    adj_matrix_unweighted = np.zeros_like(adj_matrix)
    adj_matrix_unweighted[adj_matrix.nonzero()] = 1. # type: ignore
    
    # Degree Matrix
    degree_matrix = np.zeros_like(adj_matrix)
    for edge_key in dhn_network.edges_nodes:
        (start_node, end_node) = dhn_network.edges_nodes[edge_key]
        degree_matrix[start_node, start_node] += 1
        degree_matrix[end_node, end_node] += 1
        
    # Transition Probability matrix
    transition_probability_matrix = np.matmul(adj_matrix_unweighted,np.linalg.inv(degree_matrix))

    clusters_found = []
    limit_nb_clusters = control_params[key_nb_walkers] # == number of walkers
    limit_size_clusters = control_params[key_nb_max_nodes]
    producers_from_julia = control_params[key_producers_list]

    import random
    random.seed(2)
    nb_nodes = degree_matrix.shape[0]
    list_nodes = range(nb_nodes)
    producers = [nd-1 for nd in producers_from_julia]
    while len(clusters_found) < limit_nb_clusters:
        current_cluster = []
        current_node = np.random.choice(list_nodes)
        limit_nb_steps = np.random.choice(range(2, limit_size_clusters)) # Walker steps
        for t in range(limit_nb_steps):
            prob_p = transition_probability_matrix[:,current_node]
            prob_p[producers] = 0.
            prob_p[current_cluster] = 0.0
            sum_pr = np.sum(prob_p) 
            if sum_pr == 0:
                break
            prob_p *= 1/sum_pr
            founds = random.choices(population=list_nodes, weights=prob_p)
            next_node = founds[0]
            retry = 0
            while (next_node in producers or next_node in current_cluster) and retry < 4:
                founds = random.choices(population=list_nodes, weights=prob_p)
                next_node = founds[0]
                retry += 1
            if retry == 4:
                print('Reached!')
            current_cluster.append(next_node)
            current_node = next_node
        current_cluster.sort()
        if len(current_cluster) <=1 or current_cluster in clusters_found:
            continue 
        
        clusters_found.append([ids+1 for ids in current_cluster])
            
    print(f'{len(clusters_found)} clusters selected')

    clusters_dict = {}
    clusters_dist_types = []
    clusters_dist_nb = []
    key = 0
    for cluster in clusters_found:
        cluster_key = f'{key}-c'
        type_c_str = get_cluster_types(dhn_network, cluster, cluster_key)
        type_c = type_c_str.split(' ')[-1] # type: ignore
        cluster_key = f'{key}-c'
        print(type_c_str + f' with {len(cluster)} nodes') # type: ignore
        clusters_dict[cluster_key] = cluster
        clusters_dist_types.append(type_c)
        clusters_dist_nb.append(len(cluster))
        key +=1
        
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(clusters_dist_types, bins=100)
    ax.set_xlabel('Cluster types')
    ax.set_ylabel('Distribution')
    plt.show()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(clusters_dist_nb, bins=100)
    ax.set_xlabel('Nb of nodes in the cluster')
    ax.set_ylabel('Distribution')
    plt.show()
    
    return clusters_found

def get_learned_cluster_performance(dhn: DistrictHeatingNetworkFromExcel, model_full_path: str, cluster: list[str]):
    train_x, train_y, test_x, test_y, scaller_y, scaller_x = get_data_cluster_dhn(dhn, cluster)
    
    model = load_model(model_full_path)
    predictions = scaller_y.inverse_transform(model.predict(test_x))
    reals = scaller_y.inverse_transform(test_y)
    
    inners, ingoings, outgoings, _ = dhn.get_cluster_qualities_and_identify_connecting_pipes(cluster)
    tsins = dhn.dict_physical_values['Tsin']
    trins = dhn.dict_physical_values['Trin']
    mw = dhn.dict_physical_values['mw']
    
    n_drop = 1000
    start = 0
    st_index = start+n_drop
    end_index = start+n_drop+test_x.shape[0]
    ingoing_powers = np.zeros_like(mw[0, st_index:end_index])
    ingoing_powers_preds = np.zeros_like(mw[0, st_index:end_index])
    outgoing_powers = np.zeros_like(mw[0, st_index:end_index])
    outgoing_powers_preds = np.zeros_like(mw[0, st_index:end_index])
    i_ml = 0
    for edg in ingoings:
        (st, ed) = dhn.edges_nodes[edg]
        mw_edge = mw[edg, st_index:end_index]
        ts_in = tsins[st, st_index:end_index]
        tr_out = reals[:, i_ml]
        tr_out_preds = predictions[:,i_ml]
        ingoing_powers += mw_edge*CP*(ts_in - tr_out)
        ingoing_powers_preds += mw_edge*CP*(ts_in - tr_out_preds)
        i_ml +=1
    
    for edg in outgoings:
        (st, ed) = dhn.edges_nodes[edg]
        mw_edge = mw[edg, st_index:end_index]
        tr_in = trins[ed, st_index:end_index]
        ts_out = reals[:, i_ml]
        ts_out_preds = predictions[:,i_ml]
        outgoing_powers += mw_edge*CP*(ts_out - tr_in)
        outgoing_powers_preds += mw_edge*CP*(ts_out_preds - tr_in)
        i_ml +=1
    
    total_df_powers = ingoing_powers - outgoing_powers
    total_df_powers_preds = ingoing_powers_preds - outgoing_powers_preds
    total_energy_consumed = 60*np.sum(total_df_powers)
    total_energy_consumed_preds = 60*np.sum(total_df_powers_preds)
    
    print('MAE = ',np.mean(np.abs(predictions-reals)))
    print('Mean power error relative = ',np.mean(np.abs(100*(1 - total_df_powers_preds/total_df_powers))))
    print('Energy relative error = ',100*(1 - total_energy_consumed_preds/total_energy_consumed))
    return reals, predictions, total_df_powers, total_df_powers_preds

    network_path = os.path.join(case_study_dhns_folder, f'network_{network_id}')
    topology_file = ''
    dataset_file = ''

    for dirName, subdirList, fileList in os.walk(network_path):
        for fname in fileList:
            filename, file_ext = os.path.splitext(fname)
            if filename.startswith('dataset'):
                print(f'Dataset file fond : {fname}')
                dataset_file = os.path.join(dirName,fname)
            elif file_ext == ".xlsx":
                print(f'Topology file fond : {fname}')
                topology_file = os.path.join(dirName,fname)
    
    if return_topology_file:            
        return DistrictHeatingNetworkFromExcel(topology_file, dataset_file, transient_delay_in_time_step=0, last_limit_simulation=60), topology_file
    else:
        return DistrictHeatingNetworkFromExcel(topology_file, dataset_file, transient_delay_in_time_step=0, last_limit_simulation=60)