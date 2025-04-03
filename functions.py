import os
import sys

import joblib as jbl
import math
import networkx as nx

sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir))))

from src.helpers import *
from src.utils import *

import matplotlib.pyplot as plt
set_rcParams()

import networkx as nx

import statistics as sts
from scipy.signal import correlate as cc
from scipy.signal import correlation_lags as lags

import ast
from sklearn.metrics.pairwise import euclidean_distances

works_folder = r"D:\PhD DATA\Codes & Works\Works_current"

CASE_STUDY_DHNs_FOLDER = os.path.join(works_folder, 'CASE_STUDY_DHNS_copy_here')

import networkx as nx

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
    df_ = df_.sort_values(by=x_axis)
    # if x_axis == 'type':
    #     df_ = df_.sort_values(by=['type'])
    # elif x_axis == 'size':
    #     df_ = df_.sort_values(by=['size'])
    # else:
    #     raise Exception("Only type and size are ")
    
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

def read_clusters_of_network(network_id: int, clusterings_id: int) -> dict:
    """Reads the information of already trained clusters with their composing nodes and keys

    Args:
        network_id (int): case study DHN identifier [1,2,3,4] containing the cluster
        clusterings_id (int): set of the clusters [1,2]

    Returns:
        dict: Dictionary of the clusters
    """
    with open(os.path.join(os.path.join(CASE_STUDY_DHNs_FOLDER, f'network_{network_id}'), f'considered_clusters_{clusterings_id}.json'), "r") as jsonfile:
        network_case_study_dhns = json.load(jsonfile)
        print(f'Clusterings v{clusterings_id} of network {network_id} -- succefully loaded !')
    jsonfile.close()
    return network_case_study_dhns

def get_dhn_network(network_id: int, return_topology_file = False) -> DistrictHeatingNetworkFromExcel:
    """Retrieves the information of the desired DHN and create the python object

    Args:
        network_id (int): identifier of the DHN [1,2,3,4]
        return_topology_file (bool, optional): whether to return the found topology excel file. Defaults to False.

    Returns:
        DistrictHeatingNetworkFromExcel: the DHN object
    """
    network_path = os.path.join(CASE_STUDY_DHNs_FOLDER, f'network_{network_id}')
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

def _any_flow_inversion_on_the_edge(mws: np.array):
    """Detects the edges where the sign of the MWs changes from positive to negative or vice versa.

    Args:
        mws (np.array): the mass flow rates

    Returns:
        bool: whether there is an inversion
    """
    signs = np.sign(mws[:-1]) * np.sign(mws[1:])  #  calculates the element-wise product of signs between adjacent elements. A negative product indicates a sign change (one positive and one negative).
    nt_inv = np.where(signs == -1)[0]
    return nt_inv.size > 2 # avoid anomalies of just one jump (TODO DR: see what causes that phenomenon on the julia physical simulation)

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

def compute_ann_model_errors_to_conserve_thermal_losses(cluster: list, dhn: DistrictHeatingNetworkFromExcel, mae: float) -> tuple[float] :
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