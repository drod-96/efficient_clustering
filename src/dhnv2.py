from src.constants import *

import pandas as pd
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mat73 as mat

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.cluster import KMeans

import networkx as nx

#region DHN from Excel V2
class DistrictHeatingNetworkFromExcel(object):
    """District heating network graph formulation which contains all important information from excel file
    """

    def __init__(self,
                 topology_file_path: str, 
                 dataset_file_path: str,
                 undirected = True,
                 include_nodes_coordinates_as_attributes = True,
                 index_start_for_Text = 10, 
                 index_end_for_Text = 4000,
                 last_limit_simulation = 20, # Ne pas rendre les dernieres valeurs car tend vers 0 à cause de l'arret de la simu
                 transient_delay_in_time_step = 40,  # La durée de la partie transitoire au debut
                 version_topology_file=1):
        self.topology_file_path = topology_file_path
        self.dataset_file_path = dataset_file_path
        self.undirected = undirected
        self.last_limit_simulation = last_limit_simulation
        self.topology_version = version_topology_file
        self.adjacency_matrix = []
        self.edges_nodes = {} # List edges containing start and end nodes, list of edges, useful also for GNN
        # self.nodes_attributes = [] # X vector for nodes attributes, useful only for GNN
        self.nodes_coordinates = {} # Geographical positions of the nodes
        self.include_coord = include_nodes_coordinates_as_attributes # Whether to include the coordinates
        self.nodes_colors = []
        self.producer_nodes = []
        # self.outdoor_temperatures = {}
        self.transient_delay = transient_delay_in_time_step
        self.index_start_for_Text = index_start_for_Text
        self.index_end_for_Text = index_end_for_Text+1
        self.edge_labels = {} # Useful only for the draw
        self.node_labels = {} # Node labels dict with numbers
        self.edge_features = {} # Features qu'on va ou non utiliser mais je prefere les avoir toutes en un seul endroit (n'oubliez pas d'inclure ici les inversion de flux)
        self.edge_features_v2 = {} # Better than previous one to regroup useful information
        self._generate_adjacency_matrix_and_nodes_attributes()
        # self.tss = tss
        # self.trc = 50.0
        self.inverted_edges_indices = [] # in python indices
        
        # Puis les inputs
        self.dict_physical_values = self.generate_input_dictionnary()
        
    def get_all_edges_with_inversion_fluxes(self, verbose=1):
        mw = self.dict_physical_values['mw']
        list_ = []
        for e, _ in self.edge_features_v2.iterrows():
            if self._any_flow_inversion_on_the_edge(mw[e,:]):
                list_.append(e)
        
        if verbose == 1:   
            print('Index of the edges with possible flux inversion: ', list_)
        return list_
        
    def get_inverted_edges_in_julia_indices(self):
        return [i+1 for i in self.inverted_edges_indices]

    def _compute_consumption_difference(self, cons_node_1, cons_node_2) -> float:
        """
            Computes the euclidean distance between two consumptions profiles
            
        Return:
            Mean distance
        """
        # Euclidean distance
        difference = cons_node_1 - cons_node_2
        return np.abs(np.mean(difference))

    def _compute_delay_time_pipe(self, ms, l, d):
        """Computes delay time (l must be in meter)

        Args:
            ms (_type_): _description_
            l (_type_): _description_
            d (_type_): _description_
        """
         # tau = (r^2 * pi * L * rho) / m [R.Hagg]
        r = d / 2
        return r**2 * np.pi * l * RHO / ms

    def _compute_weight_for_gnn(self, ms, l, d, h):
        """Computes the tau / tau_p (l must be in meter)

        Args:
            ms (_type_): _description_
            l (_type_): _description_
            d (_type_): _description_
            h (_type_): _description_
        """
        # tau_p = (pi * cp * rho * r^2 / (2 * h * pi * r) [Giraud + R.Hägg + Mohamed Code]
        # Donc le tau / tau_p devient juste = pi * d * L * h / cp / msfr
        # tau_by_tau_p = np.pi * d * l * h / CP / ms
        tau_by_tau_p = np.pi * d * l * h / (CP) / np.abs(ms) # Sans CP
        return np.exp(-1 * tau_by_tau_p)
    
    def compute_edges_power(self, ms, ts_in, ts_out, tr_in, tr_out):
        p_in = ms * CP * (ts_in - tr_out)
        p_out = ms * CP * (ts_out - tr_in)
        p_loss = p_in - p_out
        return p_in, p_out, p_loss
        
    def _generate_adjacency_matrix_and_nodes_attributes(self):
        """Generates the adjacency matrix in the shape of (N x N) where adj[i, j] = np.array([tau/tau_p])

        Returns:
            tuple: adj or adj, coordinates, edges list or adj, edges list or adj, coordinates
        """

        nodes_df = pd.read_excel(self.topology_file_path, sheet_name=['nodes'])['nodes']
        pipes_df = pd.read_excel(self.topology_file_path, sheet_name=['pipes'])['pipes']
        # loads_df = pd.read_excel(self.topology_file_path, sheet_name=['loads'])['loads']
        if self.topology_version == 1:
            df_conso = pd.read_excel(self.topology_file_path, sheet_name=['consumers'])['consumers']
        else:
            df_conso = pd.read_excel(self.topology_file_path, sheet_name=['consumers(area)'])['consumers(area)']
        self.consumers_infos = df_conso.copy()
        
        # outdoor_temps_df = pd.read_excel(self.topology_file_path, sheet_name=['outdoor temperature'])['outdoor temperature'] # may be useful somewhere
        # self.outdoor_temperatures = outdoor_temps_df.iloc[range(self.index_start_for_Text+self.transient_delay, self.index_end_for_Text - self.last_limit_simulation)] # as I cut in 30 points in DHN
        
        n = len(nodes_df)
        # Adjacency matrix
        adj_matrix = np.zeros(shape=(n, n), dtype=float)
        edge_index = 0
        # Attributes matrix
        att_matrix = np.zeros(shape=(n, 5 if self.include_coord else 4), dtype=float)
        # taups = []
        for edge_index, row in pipes_df.iterrows():
            start_node_index = int(row['start node']) -1
            end_node_index = int(row['end node'])-1
            self.edges_nodes[edge_index] = (start_node_index, end_node_index)

        # On enleve la premiere colonne
        pipes_df.drop(pipes_df.columns[0], axis=1)
        self.edge_features = pipes_df
        is_prod_key = 'is_prod' if self.topology_version == 1 else 'Is source'
        for node_index, row in nodes_df.iterrows():
            x = row['x']
            y = row['y']
            # demand = np.mean(loads_df[node_index+1])
            is_prod = row[is_prod_key]
            # cns = consumers_df.iloc[node_index]
            # u_factor = cns['U factor']
            # tr = row['Return temperature']
            # # te = row['Outside temperature']
            # if self.include_coord:
            #     att_matrix[node_index] = np.array([is_prod, demand, u_factor, x, y], dtype=float)
            # else:
            #     att_matrix[node_index] = np.array([is_prod, demand, u_factor], dtype=float)
            
            self.nodes_coordinates[node_index] = np.array([x, y], dtype=float)
            
            if is_prod:
                self.nodes_colors.append('tab:red')
                self.producer_nodes.append(int(node_index)+1)
            else:
                self.nodes_colors.append('tab:blue')
            # self.node_labels[node_index] = f'[{round(demand/1e6,1)}]\n[{round(tr,1)}]'
            self.node_labels[node_index] = f'[{round(0/1e3)}]'
            
        self.adjacency_matrix = adj_matrix
        # self.nodes_attributes = att_matrix
        
        if not self.undirected:
            G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.MultiDiGraph)
        else:
            G = nx.from_numpy_array(self.adjacency_matrix)
        
        self.Graph = G
    
    def plot_district_heating_graph(self, node_labels = None):
        """Plots the DH of study in graph form
        
        Returns:
            _type_: _description_
        """            
        keys = list(self.nodes_coordinates.keys())
        labels = dict([(key, int(key) + 1) for key in keys])
        
        labels_to_use = self.node_labels
        if node_labels != None:
            labels_to_use = node_labels
        plt.figure(figsize=(18, 18))
        nx.draw_networkx_labels(self.Graph, pos=self.nodes_coordinates, labels=labels_to_use, horizontalalignment="left", verticalalignment='bottom', font_weight=2)
        nx.draw_networkx_edge_labels(self.Graph, pos=self.nodes_coordinates, edge_labels=self.edge_labels)
        nx.draw(self.Graph, pos=self.nodes_coordinates, node_color=self.nodes_colors)
        plt.show()
        
    def _treat_convergence_problem_mass_flow_rates(self, mw):
        mean_value = np.max(np.abs(mw))
        mw[np.abs(mw) < 0.01*mean_value] = 0.0
        return mw
        
    def _any_flow_inversion_on_the_edge(self, mws):
        """Detects the edges where the sign of the MWs changes from positive to negative or vice versa.

        Args:
            mws (np.array): the mass flow rates

        Returns:
            bool: whether there is an inversion
        """
        signs = np.sign(mws[:-1]) * np.sign(mws[1:])  #  calculates the element-wise product of signs between adjacent elements. A negative product indicates a sign change (one positive and one negative).
        nt_inv = np.where(signs == -1)[0]
        return nt_inv.size > 2 # avoid anomalies of just one jump (TODO DR: see what causes that phenomenon on the julia physical simulation)
            
    def generate_input_dictionnary(self):
        """Generates the input dictionnary
            
            Some of the edges have been arbitrary chosen to be in opposite so we have "full" negative flow rate, 
            we must change this orientation by multipliying by - 1 the mass rate

        Args:
            matlab_file (str): the matlab file containing the data from physical simulations
            inverse_edges (list, optional): the inverse edges by default in the district heating of study. Defaults to [1, 2, 17].

        Returns:
            dict: the dictionnary containing the values
        """
        mat_file = self.dataset_file_path
        cut = self.transient_delay
        limit = self.last_limit_simulation
        
        matlab_variables = mat.loadmat(mat_file, use_attrdict=True)
        
        # Demands loads 
        consumptions = matlab_variables['load'][:,cut:-limit]
        key_topology = 'topology_load'
        if key_topology not in matlab_variables:
            key_topology = 'load'
        if key_topology not in matlab_variables:
            key_topology = 'real_Pc'
        forecast_demands = matlab_variables[key_topology][:,cut:-limit]

        # Supply temperature at the center of each node (combination from all suppliers) = temperature supplied to next nodes
        nodes_supply_temperature_dynamic = matlab_variables['ts'][:,cut:-limit] if self.topology_version == 1 else matlab_variables['tsDynamic'][:,cut:-limit]
        # Return temperature at the center of each node (combination from all returners) = temperature returned to previous nodes
        nodes_return_temperature_dynamic = matlab_variables['tr'][:,cut:-limit] if self.topology_version == 1 else matlab_variables['trDynamic'][:,cut:-limit]
        
        # Temperatures at the exit of the pipes
        pipes_tsin = matlab_variables['tsin'][:,cut:-limit]
        pipes_tsout = matlab_variables['tsout'][:,cut:-limit]
        pipes_trin = matlab_variables['trin'][:,cut:-limit]
        pipes_trout = matlab_variables['trout'][:,cut:-limit]
        mass_rates_in_pipes = matlab_variables['mw'][:,cut:-limit]
        consumptions_ms_rates = matlab_variables['mc'][:,cut:-limit]
        if 'ms' in matlab_variables:
            sources_ms_rates = matlab_variables['ms'][:,cut:-limit]
        else:
            sources_ms_rates = np.zeros_like(mass_rates_in_pipes)
        # sources_ms_rates = matlab_variables['ms'][:,cut:-limit]
        sources_tss = np.zeros_like(consumptions_ms_rates)
        if 'tss' in matlab_variables:
            sources_tss = matlab_variables['tss'][:,cut:limit]
            
        sources_ps = np.zeros_like(consumptions_ms_rates)
        if 'Ps' in matlab_variables:
            sources_ps = matlab_variables['Ps'][:,cut:limit]
        # consumer_deficits = matlab_variables['deficit'][:,cut:-limit]
        # consumer_surplus = matlab_variables['surplus'][:,cut:-limit]
        charges_at_nodes = np.zeros_like(consumptions)
        
        taus = []
        # Mass rates
        e_features = dict()
        for edge_idx in self.edges_nodes:
            msw = mass_rates_in_pipes[edge_idx, :]
            msw = self._treat_convergence_problem_mass_flow_rates(msw)
            (st, ed) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx] # not rearranged in inverted edges because it is hard to remodify pandas dataframe
            tau_edge = self._compute_delay_time_pipe(np.mean(np.abs(msw)), row['length']*1e3, row['Diameter'])
            taus.append(tau_edge)
            nb_ = ed
            # if self._any_flow_inversion_on_the_edge(msw):
            if all(item <= 0.01 for item in msw):
                mass_rates_in_pipes[edge_idx, :] = np.abs(msw)
                # inverse nodes
                print(f'Edge {edge_idx+1} is inversed due to flux inversion!')
                self.edges_nodes[edge_idx] = (ed, st)
                nb_ = st
                self.inverted_edges_indices.append(edge_idx)

            e_features[edge_idx] = {
                'start node': self.edges_nodes[edge_idx][0],
                'end node': self.edges_nodes[edge_idx][1],
                'length': row['length'] * 1000.0,
                'diameter': row['Diameter'],
                'delay_time': tau_edge,
                'velocity': np.mean(np.abs(msw)) / (RHO * np.pi * (row['Diameter']/2)**2)
            }
            
            charges_at_nodes[nb_,:] = mass_rates_in_pipes[edge_idx, :] * CP * (nodes_supply_temperature_dynamic[nb_,:] - nodes_return_temperature_dynamic[nb_,:])
            # CHANGER ICI SI JE VEUX CONSIDERER DES GRAPHES ORIENTES
            self.adjacency_matrix[st, ed] = self._compute_weight_for_gnn(np.mean(np.abs(msw)), row['length']*1e3, row['Diameter'], row['h'])
            self.adjacency_matrix[ed, st] = self.adjacency_matrix[st, ed]
            self.edge_labels[(st, ed)] = round(self.adjacency_matrix[st, ed],2)
        
        self.edge_features_v2 = pd.DataFrame.from_records(e_features).T
        self.edge_features.insert(loc=len(self.edge_features.columns), column='Delay time', value=np.array(taus))
        
        dict_inputs = {
            'Demands': np.abs(consumptions), # car en W #
            'Real_Pc': consumptions,
            'Forecast_demands': forecast_demands, # verifier
            'mw': mass_rates_in_pipes,
            'Tr_node': nodes_return_temperature_dynamic,
            'Ts_node': nodes_supply_temperature_dynamic,
            'Chr_node': charges_at_nodes,
            'Trc': matlab_variables['trcs'],
            'Tsin': pipes_tsin,
            'Tsout': pipes_tsout,
            'Trin': pipes_trin,
            'Trout': pipes_trout,
            'mc': consumptions_ms_rates,
            'ms': sources_ms_rates,
            'tss': sources_tss,
            'Ps': sources_ps,
            # 'surplus': consumer_surplus,
            # 'deficit': consumer_deficits,
        }
        
        return dict_inputs
      
    def interpolate(self, array2d):
        new_array2d = np.zeros_like(array2d)
        for j in range(array2d.shape[0]):
            prev_array = array2d[j,:]
            new_array1d = np.zeros_like(prev_array)
            new_array1d[0] = prev_array[0]
            n_dynamics = int(new_array1d.shape[0] / 60)
            for ii in range(1, n_dynamics+1):
                prev_st = (ii-1)*60
                if ii == n_dynamics:
                    next_st = ii*60-1
                else:
                    next_st = ii*60
                prev_value = prev_array[prev_st]
                new_value = prev_array[next_st]
                array_inter = np.interp(range(prev_st, next_st), [prev_st, next_st], [prev_value, new_value])
                new_array1d[prev_st:next_st] = array_inter
            new_array2d[j,:] = new_array1d
        return new_array2d

    def get_cluster_signals_smoothness(self, cluster_of_nodes: list[int]):
        """Computes the smoothness of the cluster signals [X^s, X^r, X^d] = X
        Normalized local smoothness = (X^T . nL . X) / (X^T.X) with normalized laplacian nL = I - D^{-1/2}.W.D^{1/2} = I - D^{-1}.W
        Normalized laplacian normL = 1 - (1/sqrt(D))W(1/sqrt(D))
        # Normalized local smoothness (also smoothness index [Local Smoothness of Graph Signals, M. Dakovic et al., 2019])

        Args:
            cluster_of_nodes (list[int]): List of nodes inside the cluster
            
        Returns:
            array[int]: list of smoothness over time
        """
        
        dict_inputs = self.dict_physical_values
        inner_edges, in_going_edges, out_going_edges, _ = self.get_cluster_qualities_and_identify_connecting_pipes(cluster_of_nodes) # at time = 0
        demands = dict_inputs['Demands'][:, 0:60000].copy()
        ms_rates = dict_inputs['mw'][:, 0:60000].copy()
        nodes_ts = dict_inputs['Ts_node'][:, 0:60000].copy()
        nodes_tr = dict_inputs['Tr_node'][:, 0:60000].copy()
        (nb_nodes, length_times) = demands.shape
        size_time_length = length_times
        x_series = np.zeros(shape=(nb_nodes, 4, size_time_length)) # N x 3 x TimeSteps
        weight_series = np.zeros(shape=(nb_nodes, nb_nodes, size_time_length)) # N x N x TimeSteps
        
        # Computes nodes local degree
        inner_nodes_degree = {}
        for edge_idx in inner_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            if sti not in inner_nodes_degree:
                inner_nodes_degree[sti] = 1
            else :
                inner_nodes_degree[sti] += 1
                
            if endi not in inner_nodes_degree:
                inner_nodes_degree[endi] = 1
            else :
                inner_nodes_degree[endi] += 1
        
        # Constructions of X matrix
        nodes_associated_index_in_x = {}
        asso_index = 0
        for node_idx in inner_nodes_degree.keys():
            x_series[asso_index,0,:] = nodes_ts[node_idx,:]
            x_series[asso_index,1,:] = nodes_tr[node_idx,:]
            x_series[asso_index,2,:] = demands[node_idx,:]
            nodes_associated_index_in_x[node_idx] = asso_index
            asso_index += 1
        
        for edge_idx in inner_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:]))
        
            d_i = inner_nodes_degree[sti]
            d_j = inner_nodes_degree[endi]

            weight_series[nodes_associated_index_in_x[sti], nodes_associated_index_in_x[endi], :] = -1 * weight / (d_i*d_j)
            weight_series[nodes_associated_index_in_x[endi], nodes_associated_index_in_x[sti], :] = -1 * weight / (d_i*d_j)
            weight_series[nodes_associated_index_in_x[sti], nodes_associated_index_in_x[sti], :] = np.ones_like(weight)
            weight_series[nodes_associated_index_in_x[endi], nodes_associated_index_in_x[endi], :] = np.ones_like(weight)
        
         # Compute smoothness normalized
        temporal_smoothness_index = []
        for t in range(x_series.shape[2]):
            x_ = x_series[:,:,t]
            w_ = weight_series[:,:,t]
            w_x_ = np.matmul(w_, x_)
            x_w_x_ = np.matmul(x_.T, w_x_)
            x_w_x_n = x_w_x_ * (1/(np.linalg.norm(x_, ord='fro')**2)) # Default = Frobenius norm
            # sm_index = np.trace(x_w_x_n) / np.linalg.norm(weight_series[:,:,t])
            sm_index = np.trace(x_w_x_n)
            
            if sm_index == np.nan or sm_index == np.inf or np.abs(sm_index) > 1e4:
                sm_index = 0.0
            temporal_smoothness_index.append(sm_index)
            
        return temporal_smoothness_index
    
    def compute_cluster_metrics(self, cluster_of_nodes: list[int]) -> dict:
        """This function is created to generate only topological metrics of the clusters, references are included here or elsewhere in the project
        
        Args:
            cluster_of_nodes (list[int]): the list of nodes composing the cluster

        Returns:
            dict: dictionary containing the topological metrics
        """
        
        inner_edges, in_going_edges, out_going_edges, qualities = self.get_cluster_qualities_and_identify_connecting_pipes(cluster_of_nodes) # at time = 0
        
        list_edges = []
        for e in inner_edges:
            (u,v) = self.edges_nodes[e]
            list_edges.append((u,v))
        g = nx.DiGraph(list_edges)
        try:
            nx.find_cycle(g, orientation='ignore')
            is_loop = True 
        except Exception as ex:
            is_loop = False
        
        cluster_physical_metrics = {
                        
            'density': qualities['density'],
            'scaled_density': qualities['scaled_density'],
            'nodes_avg_int_indegree': qualities['nodes_mean_internal_indegree'],
            'nodes_max_int_indegree': qualities['nodes_max_internal_indegree'],
            'nodes_avg_int_outdegree': qualities['nodes_mean_internal_outdegree'],
            'nodes_max_int_outdegree': qualities['nodes_max_internal_outdegree'],
            
            'avg_odf': qualities['average_outdegree_fraction'],
            'max_odf': qualities['max_outdegree_fraction'],
            'avg_idf': qualities['average_indegree_fraction'],
            'max_idf': qualities['max_indegree_fraction'],
            
            'cut_size': qualities['cut_size'],
            'cut_ratio': qualities['cut_ratio'],
            'conductance': qualities['conductance'],
            'cluster_dimater': qualities['max_diameter'],
            
            'cluster_has_loop': is_loop,
        }
        
        return cluster_physical_metrics
        
    def get_cluster_trial_metrics(self, cluster_of_nodes: list[int]) -> dict:
        """Generates a dictionnary with all the metrics of the clusters without time dependency
        Including:  
                    return energy loss
                    supply energy loss
                    
                    equivalent delta trc
                    
                    demands evolutions similarities => 'demands_sim' = {'min', 'max', 'mean'}
                    total pipes surface => 'total_surface_pipes'
                    number of consumer nodes => 'nb_consumers'
                    number of pipes (incoming+inner+outgoing) => 'nb_pipes_in_ext'

        Args:
            cluster_of_nodes (list[int]): The cluster of nodes
            
        Returns:
            dict: metrics so far tried
        """
        
        dict_inputs = self.dict_physical_values
        inner_edges, in_going_edges, out_going_edges, qualities = self.get_cluster_qualities_and_identify_connecting_pipes(cluster_of_nodes) # at time = 0
        
        demands = dict_inputs['Demands'][:, :80000].copy()
        ms_rates = dict_inputs['mw'][:, :80000].copy()
        pipes_tsin = dict_inputs['Tsin'][:, :80000].copy()
        pipes_tsout = dict_inputs['Tsout'][:, :80000].copy()
        pipes_trin = dict_inputs['Trin'][:, :80000].copy()
        pipes_trout = dict_inputs['Trout'][:, :80000].copy()
        nodes_trcs = dict_inputs['Trc'].copy()
        
        power_ingoing_supply = []
        power_outgoing_supply = []
        
        power_ingoing_return = []
        power_outgoing_return = []
        
        stored_power_supply = []
        stored_power_return = []
        
        total_stored_lost_energy_through_pipes = 0
        list_of_trcs = []
        
        equivalent_dtes_weighted_demands = []
        sum_demands = []
        
        total_inside_pipes_surface = 0
        total_outside_pipes_surface = 0
        
        mean_h_outside_pipes = 0
        mean_h_inside_pipes = 0
        
        outside_edges_inversed_steps = 0
        inside_edges_inversed_steps = 0
        
        mean_distance_consumptions_profiles = 0
        
        # We compute for all time steps
        min_trc = 1000
        for edge_idx in inner_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            
            mean_distance_consumptions_profiles += self._compute_consumption_difference(demands[sti,:], demands[endi,:])
            min_endi_sti = np.min([nodes_trcs[sti], nodes_trcs[endi]])
            if min_trc > min_endi_sti:
                min_trc = min_endi_sti
        
        mean_distance_consumptions_profiles /= len(inner_edges)
        # intracluster_demands_distance_diameter = self.compute_intracluster_diameter(cluster_of_nodes)
        for node in cluster_of_nodes:
            node_idx = int(node-1)
            demand_node = demands[node_idx,:]
            list_of_trcs.append(nodes_trcs[node_idx])
            if len(sum_demands) == 0:
                equivalent_dtes_weighted_demands = np.abs(nodes_trcs[node_idx] - min_trc)*demand_node
                sum_demands = demand_node
            else:
                equivalent_dtes_weighted_demands += np.abs(nodes_trcs[node_idx] - min_trc)*demand_node
                sum_demands += demand_node
            
        equivalent_dtes_weighted_demands /= sum_demands
        
        ingoing_tsins = []
        outgoing_trouts = []
        for edge_idx in in_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            total_outside_pipes_surface += np.pi * lng * d
            mean_h_outside_pipes += h

            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            
            p_in = ms_rates[edge_idx, :] * CP * (tsin_edge - trout_edge)
            p_out = ms_rates[edge_idx, :] * CP * (tsout_edge - trin_edge)
            
            total_stored_lost_energy_through_pipes += np.sum(p_in - p_out) * 60
            
            ingoing_tsins.append(tsin_edge)
            outgoing_trouts.append(trout_edge)
            
            average_ts_over_pipes = 0.5*(tsin_edge + tsout_edge)
            average_tr_over_pipes = 0.5*(trin_edge + trout_edge)

            stored_ts = []
            stored_tr = []
            
            outside_edges_inversed_steps += np.sum(ms_rates[edge_idx,:] < 0) / len(ms_rates[edge_idx,:])
            
            vlw = (np.pi*(d/2)*(d/2)*lng)
            for t in range(1, len(average_tr_over_pipes)):
                stored_ts.append(RHO*CP*vlw*(average_ts_over_pipes[t] - average_ts_over_pipes[t-1]))
                stored_tr.append(RHO*CP*vlw*(average_tr_over_pipes[t] - average_tr_over_pipes[t-1]))
                
            if len(power_ingoing_supply) == 0:
                power_ingoing_supply = ms_rates[edge_idx,:]*CP*(tsin_edge)*60
                power_outgoing_supply = ms_rates[edge_idx,:]*CP*(tsout_edge)*60
                stored_power_supply = np.array(stored_ts)
                
                power_ingoing_return = ms_rates[edge_idx,:]*CP*(trin_edge)*60
                power_outgoing_return = ms_rates[edge_idx,:]*CP*(trout_edge)*60
                stored_power_return = np.array(stored_tr)
                
            else:
                power_ingoing_supply += ms_rates[edge_idx,:]*CP*(tsin_edge)*60
                power_outgoing_supply += ms_rates[edge_idx,:]*CP*(tsout_edge)*60
                stored_power_supply += np.array(stored_ts)
                
                power_ingoing_return += ms_rates[edge_idx,:]*CP*(trin_edge)*60
                power_outgoing_return += ms_rates[edge_idx,:]*CP*(trout_edge)*60
                stored_power_return += np.array(stored_tr)

        for edge_idx in inner_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            total_inside_pipes_surface += np.pi * d * lng
            mean_h_inside_pipes += h
            
            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            
            average_ts_over_pipes = 0.5*(tsin_edge + tsout_edge)
            average_tr_over_pipes = 0.5*(trin_edge + trout_edge)
            
            p_in = ms_rates[edge_idx, :] * CP * (tsin_edge - trout_edge)
            p_out = ms_rates[edge_idx, :] * CP * (tsout_edge - trin_edge)
            
            total_stored_lost_energy_through_pipes += np.sum(p_in - p_out) * 60

            inside_edges_inversed_steps += np.sum(ms_rates[edge_idx,:] < 0) / len(ms_rates[edge_idx,:])
            
            stored_ts = []
            stored_tr = []
            
            vlw = (np.pi*(d/2)*(d/2)*lng)
            for t in range(1, len(average_tr_over_pipes)):
                stored_ts.append(RHO*CP*vlw*(average_ts_over_pipes[t] - average_ts_over_pipes[t-1]))
                stored_tr.append(RHO*CP*vlw*(average_tr_over_pipes[t] - average_tr_over_pipes[t-1]))
            
            power_ingoing_supply += ms_rates[edge_idx,:]*CP*(tsin_edge)*60
            power_outgoing_supply += ms_rates[edge_idx,:]*CP*(tsout_edge)*60
            stored_power_supply += np.array(stored_ts)
            
            power_ingoing_return += ms_rates[edge_idx,:]*CP*(trin_edge)*60
            power_outgoing_return += ms_rates[edge_idx,:]*CP*(trout_edge)*60
            stored_power_return += np.array(stored_tr)
            
        inside_edges_inversed_steps /= len(inner_edges)
        ingoing_trins = []
        outgoing_tsouts = []
        for edge_idx in out_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            total_outside_pipes_surface += np.pi * lng * d
            mean_h_outside_pipes += h
            
            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            
            ingoing_trins.append(trin_edge)
            outgoing_tsouts.append(tsout_edge)
            
            average_ts_over_pipes = 0.5*(tsin_edge + tsout_edge)
            average_tr_over_pipes = 0.5*(trin_edge + trout_edge)
            
            p_in = ms_rates[edge_idx, :] * CP * (tsin_edge - trout_edge)
            p_out = ms_rates[edge_idx, :] * CP * (tsout_edge - trin_edge)
            
            total_stored_lost_energy_through_pipes += np.sum(p_in - p_out) * 60
            
            outside_edges_inversed_steps += np.sum(ms_rates[edge_idx,:] < 0) / len(ms_rates[edge_idx,:])
            
            stored_ts = []
            stored_tr = []
            
            vlw = (np.pi*(d/2)*(d/2)*lng)
            for t in range(1, len(average_tr_over_pipes)):
                stored_ts.append(RHO*CP*vlw*(average_ts_over_pipes[t] - average_ts_over_pipes[t-1]))
                stored_tr.append(RHO*CP*vlw*(average_tr_over_pipes[t] - average_tr_over_pipes[t-1]))
            
            power_ingoing_supply += ms_rates[edge_idx,:]*CP*(tsin_edge)*60
            power_outgoing_supply += ms_rates[edge_idx,:]*CP*(tsout_edge)*60
            stored_power_supply += np.array(stored_ts)
            
            power_ingoing_return += ms_rates[edge_idx,:]*CP*(trin_edge)*60
            power_outgoing_return += ms_rates[edge_idx,:]*CP*(trout_edge)*60
            stored_power_return += np.array(stored_tr)
            
        # nb_edges = len(inner_edges) + len(out_going_edges) + len(in_going_edges)  
        ext_len = (len(in_going_edges) + len(out_going_edges))
        
        mean_h_inside_pipes /= len(inner_edges)
        mean_h_outside_pipes /= ext_len
        outside_edges_inversed_steps /= ext_len
        
        # Compute the difference between ingoing temperature signals and outgoing temperature signals
        tr_sgl_diff = 0
        for trin_sgl in ingoing_trins:
            for trout_sgl in outgoing_trouts:
                tr_sgl_diff += np.abs(np.mean(trin_sgl - trout_sgl))
                
        ts_sgl_diff = 0
        for tsin_sgl in ingoing_tsins:
            for tsout_sgl in outgoing_tsouts:
                ts_sgl_diff += np.abs(np.mean(tsin_sgl - tsout_sgl))
        
        cluster_physical_metrics = {
            'total_ingoing_energy': np.sum(power_ingoing_supply) + np.sum(power_ingoing_return),
            'total_outgoing_energy': np.sum(power_outgoing_supply) + np.sum(power_outgoing_return),
            'total_stored_energy': np.sum(stored_power_supply) + np.sum(stored_power_return),
            
            'total_balance_energy_through_pipes': total_stored_lost_energy_through_pipes, 
            
            'delta_trc_mean': np.mean(equivalent_dtes_weighted_demands),
            'delta_trc_std': np.std(equivalent_dtes_weighted_demands),
            'trc_std': np.std(np.array(list_of_trcs)),
            
            'difference_tr_signals': tr_sgl_diff,
            'difference_ts_signals': ts_sgl_diff,
            
            'consumption_profiles_ed_distance': mean_distance_consumptions_profiles,
            
            'type_cluster': f'{len(in_going_edges)}-{len(out_going_edges)}',
            'total_surface_pipes_inside': total_inside_pipes_surface,
            'total_surface_pipes_outside': total_outside_pipes_surface,
            
            'mean_conv_coeff_pipes_inside': mean_h_inside_pipes,
            'mean_conv_coeff_pipes_outside': mean_h_outside_pipes,
            
            'nb_consumers': len(cluster_of_nodes),
            'nb_pipes_in_int': len(inner_edges),
            
            'nb-ingoing sp pipes': len(in_going_edges),
            'nb-ingoing rt pipes': len(out_going_edges),
            'nb_pipes_in_ext': len(in_going_edges) + len(out_going_edges),
            
            'density_cluster': qualities['density'],
            'cut_ratio': qualities['cut_ratio'],
            'conductance': qualities['conductance'],
            'cluster_dimater': qualities['max_diameter'],
            
            'inverted_msf_time_steps_inside_edges': inside_edges_inversed_steps,
            'inverted_msf_time_steps_outside_edges': outside_edges_inversed_steps,
        }
        
        cluster_physical_metrics['total_lost_energy'] = cluster_physical_metrics['total_ingoing_energy'] - cluster_physical_metrics['total_outgoing_energy'] - cluster_physical_metrics['total_stored_energy']
        
        smoothness_temporal_indices = self.get_cluster_signals_smoothness(cluster_of_nodes)
        cluster_physical_metrics['norm_smoothness_mean'] = np.mean(np.array(smoothness_temporal_indices))
        cluster_physical_metrics['norm_smoothness_std'] = np.std(np.array(smoothness_temporal_indices))
        
        return cluster_physical_metrics

    def generate_sequential_input_data_ecos(self, cluster_of_nodes: list):
        
        dict_inputs = self.dict_physical_values
        _, in_going_edges, out_going_edges, _ = self.get_cluster_qualities_and_identify_connecting_pipes(cluster_of_nodes) # at time = 0
        
        demands = dict_inputs['Forecast_demands'].copy()
        ms_rates = dict_inputs['mw'].copy()
        pipes_tsin = dict_inputs['Tsin'].copy()
        pipes_tsout = dict_inputs['Tsout'].copy()
        pipes_trin = dict_inputs['Trin'].copy()
        pipes_trout = dict_inputs['Trout'].copy()
        
        df_input_features = pd.DataFrame()
        iii = 0
            
        df_outputs = pd.DataFrame()
        ooo = 0
        
        for el in cluster_of_nodes:
            node_index = int(el)-1
            demands_node = demands[node_index,:].T
            df_input_features.insert(loc=iii, column=f'Demand {node_index+1}', value=demands_node)
            iii += 1
        
        for edge_idx in in_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']

            tsin_edge = pipes_tsin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            
            weight = np.exp(-1*(h*lng*d/CP/np.abs(ms_rates[edge_idx,:].T)))
            
            ts_in = tsin_edge.T
            df_input_features.insert(loc=iii, column=f'Tsin_pipe{edge_idx}->{sti+1}', value=ts_in*weight)
            iii +=1
            # df_input_features.insert(loc=iii, column=f'Mwin_pipe{edge_idx}->{sti+1}', value=np.abs(ms_rates[edge_idx,:]))
            # iii +=1
            
            tr_out = trout_edge.T  
            df_outputs.insert(loc=ooo, column=f'Trout_pipe{edge_idx}->{sti+1}', value=tr_out)
            ooo += 1
            
        for edge_idx in out_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            
            weight = np.exp(-1*(h*lng*d/CP/np.abs(ms_rates[edge_idx,:].T)))
              
            ts_out = tsout_edge.T
            df_outputs.insert(loc=ooo, column=f'Tsout_pipe{edge_idx}->{endi+1}', value=ts_out)
            ooo += 1

            tr_in = trin_edge.T
            df_input_features.insert(loc=iii, column=f'Trin_pipe{edge_idx}->{endi+1}', value=tr_in*weight)
            iii += 1
            
            # df_input_features.insert(loc=iii, column=f'Mwin_pipe{edge_idx}->{sti+1}', value=np.abs(ms_rates[edge_idx,:]))
            # iii +=1

        return df_input_features, df_outputs
   
    def generate_sequential_input_data_v5(self, cluster_of_nodes: list):
        
        dict_inputs = self.dict_physical_values
        _, in_going_edges, out_going_edges, _ = self.get_cluster_qualities_and_identify_connecting_pipes(cluster_of_nodes) # at time = 0
        
        demands = dict_inputs['Demands'].copy()
        ms_rates = dict_inputs['mw'].copy()
        pipes_tsin = dict_inputs['Tsin'].copy()
        pipes_tsout = dict_inputs['Tsout'].copy()
        pipes_trin = dict_inputs['Trin'].copy()
        pipes_trout = dict_inputs['Trout'].copy()
        
        df_input_features = pd.DataFrame()
        iii = 0
            
        df_outputs = pd.DataFrame()
        ooo = 0
        
        for el in cluster_of_nodes:
            node_index = int(el)-1
            demands_node = demands[node_index,:].T
            df_input_features.insert(loc=iii, column=f'Demand {node_index+1}', value=demands_node)
            iii += 1
        
        for edge_idx in in_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']

            tsin_edge = pipes_tsin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:].T))
            
            ts_in = tsin_edge.T
            df_input_features.insert(loc=iii, column=f'Tsin_pipe{edge_idx}->{sti+1}', value=ts_in*weight)
            iii +=1
            
            tr_out = trout_edge.T  
            df_outputs.insert(loc=ooo, column=f'Trout_pipe{edge_idx}->{sti+1}', value=tr_out)
            ooo += 1
            
        for edge_idx in out_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:].T))
              
            ts_out = tsout_edge.T
            df_outputs.insert(loc=ooo, column=f'Tsout_pipe{edge_idx}->{endi+1}', value=ts_out)
            ooo += 1

            tr_in = trin_edge.T
            df_input_features.insert(loc=iii, column=f'Trin_pipe{edge_idx}->{endi+1}', value=tr_in*weight)
            iii += 1

        return df_input_features, df_outputs
    
    def generate_sequential_input_data_v5_v1(self, cluster_of_nodes: list):
        
        dict_inputs = self.dict_physical_values
        inner_edges, in_going_edges, out_going_edges, _ = self.get_cluster_qualities_and_identify_connecting_pipes(cluster_of_nodes) # at time = 0
        
        demands = dict_inputs['Forecast_demands'].copy()
        ms_rates = dict_inputs['mw'].copy()
        pipes_tsin = dict_inputs['Tsin'].copy()
        pipes_tsout = dict_inputs['Tsout'].copy()
        pipes_trin = dict_inputs['Trin'].copy()
        pipes_trout = dict_inputs['Trout'].copy()
        nodes_ts = dict_inputs['Ts_node'].copy()
        nodes_tr = dict_inputs['Tr_node'].copy()
        
        mean_diameter = 0
        mean_length = 0
        mean_mwf = 0
        
        df_input_features = pd.DataFrame()
        iii = 0
            
        df_outputs = pd.DataFrame()
        ooo = 0
        
        df_temperatures_ins = pd.DataFrame() # in going temperatures to pipes to compute loss performance
        lll = 0
        
        treated_nodes = []
        array_to_copy = np.zeros_like(np.zeros_like(pipes_trin[0,:]))
        total_loss_storage_supply = np.zeros_like(array_to_copy)
        total_loss_storage_return = np.zeros_like(array_to_copy)
        total_weights = np.ones_like(array_to_copy)
        total_dts_supply = 0
        total_dts_return = 0
        spatial_smoothness_demand = 0 # W_i,j * (D_i - D_j)^2 for all inner i,j and all time step t (sur les noeuds à aggreger)
        spatial_smoothness_ts = 0 # sur aussi les pipes adjacentes
        spatial_smoothness_tr = 0
        outside_dts_supply = 0
        outside_dts_return = 0
        
        for el in cluster_of_nodes:
            node_index = int(el)-1
            demands_node = demands[node_index,:].T
            df_input_features.insert(loc=iii, column=f'Demand {node_index+1}', value=demands_node)
            iii += 1
        
        # Laplacian L = D - W
        # Normalized laplacian normL = 1 - (1/sqrt(D))W(1/sqrt(D))
        # Local smoothness = X^T.normL.X
        # Normalized local smoothness (also smoothness index [Local Smoothness of Graph Signals, M. Dakovic et al., 2019])
        # = (X^T.normL.X)/(X^T.X)
        
        # We compute for all time steps
        inner_nodes_degree = {}
        for edge_idx in inner_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            if sti not in inner_nodes_degree:
                inner_nodes_degree[sti] = 1
            else :
                inner_nodes_degree[sti] += 1
                
            if endi not in inner_nodes_degree:
                inner_nodes_degree[endi] = 1
            else :
                inner_nodes_degree[endi] += 1
        
        x_series = np.zeros(shape=(len(inner_nodes_degree.keys()), 4, len(demands_node))) # N x NbFeatures x TimeSteps
        weight_series = np.zeros(shape=(len(inner_nodes_degree.keys()), len(inner_nodes_degree.keys()), len(demands_node)))      
        nodes_associated_index_in_x = {}
        asso_index = 0
        for node_idx in inner_nodes_degree.keys():
            demand_node = demands[node_idx,:]
            ts_node = nodes_ts[node_idx,:]
            tr_node = nodes_tr[node_idx,:]
            ps_node = np.zeros_like(ts_node)
            x_series[asso_index,0,:] = ts_node
            x_series[asso_index,1,:] = tr_node
            x_series[asso_index,2,:] = demand_node
            x_series[asso_index,3,:] = ps_node
            nodes_associated_index_in_x[node_idx] = asso_index
            asso_index += 1
        
        for edge_idx in in_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            mean_diameter += d
            mean_length += lng

            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            ts_node_i = nodes_ts[sti,:]
            ts_node_j = nodes_ts[endi,:]
            tr_node_i = nodes_tr[sti,:]
            tr_node_j = nodes_tr[endi,:]
            demand_node_i = demands[sti,:]
            demand_node_j = demands[endi,:]
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:].T))
            mean_mwf += np.mean(ms_rates[edge_idx,:])
            
            total_loss_storage_supply += -1*ms_rates[edge_idx,:]*CP*(tsout_edge - tsin_edge)
            total_loss_storage_return += -1*ms_rates[edge_idx,:]*CP*(trout_edge - trin_edge)
            total_dts_supply += np.mean(np.abs(tsin_edge - tsout_edge))
            total_dts_return += np.mean(np.abs(trin_edge - trout_edge))
            outside_dts_supply += (weight.dot(np.abs(ts_node_i - ts_node_j))) / len(weight)
            outside_dts_return += (weight.dot(np.abs(tr_node_i - tr_node_j))) / len(weight)
            total_weights *= weight
            
            ts_in = tsin_edge.T
            df_input_features.insert(loc=iii, column=f'Tsin_pipe{edge_idx}->{sti+1}', value=ts_in*weight)
            iii +=1
            
            tr_out = trout_edge.T  
            df_outputs.insert(loc=ooo, column=f'Trout_pipe{edge_idx}->{sti+1}', value=tr_out)
            ooo += 1
            
            tr_in = trin_edge.T
            df_temperatures_ins.insert(loc=lll, column=f'Trin_pipe{edge_idx}->{sti+1}', value=tr_in)
            lll +=1
            
            # Smoothness
            # spatial_smoothness_tr += weight*(trout_edge - trin_edge)
            # spatial_smoothness_ts += weight*(tsout_edge - tsin_edge)
            
        demands_evolution_difference_mean = 0
        ts_difference_mean = 0
        tr_difference_mean = 0
        for edge_idx in inner_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            mean_diameter += d
            mean_length += lng
            
            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            ts_node_i = nodes_ts[sti,:]
            ts_node_j = nodes_ts[endi,:]
            tr_node_i = nodes_tr[sti,:]
            tr_node_j = nodes_tr[endi,:]
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:]))
            mean_mwf += np.mean(ms_rates[edge_idx,:])
            
            total_loss_storage_supply += -1*ms_rates[edge_idx,:]*CP*(tsout_edge - tsin_edge)
            total_loss_storage_return += -1*ms_rates[edge_idx,:]*CP*(trout_edge - trin_edge)
            total_weights *= weight
            total_dts_supply += np.mean(np.abs(tsin_edge - tsout_edge))
            total_dts_return += np.mean(np.abs(trin_edge - trout_edge))
            
            # Smoothness
            demand_sti = self.get_weighted_difference_signal(demands[sti,:])
            demand_endi = self.get_weighted_difference_signal(demands[endi,:])
            demands_evolution_difference_mean += np.mean(np.abs(demand_sti - demand_endi))
            ts_difference_mean += np.mean(np.abs(ts_node_j - ts_node_i))
            tr_difference_mean += np.mean(np.abs(tr_node_j - tr_node_i))
            
            d_i = inner_nodes_degree[sti]
            d_j = inner_nodes_degree[endi]
            weight_series[nodes_associated_index_in_x[sti], nodes_associated_index_in_x[sti], :] = np.ones_like(weight)
            weight_series[nodes_associated_index_in_x[endi], nodes_associated_index_in_x[endi], :] = np.ones_like(weight)
            weight_series[nodes_associated_index_in_x[sti], nodes_associated_index_in_x[endi], :] = weight / (d_i*d_j)
            weight_series[nodes_associated_index_in_x[endi], nodes_associated_index_in_x[sti], :] = weight / (d_i*d_j)
        
        demands_evolution_difference_mean /= len(cluster_of_nodes)
        tr_difference_mean /= len(cluster_of_nodes)
        ts_difference_mean /= len(cluster_of_nodes)
          
        for edge_idx in out_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            mean_diameter += d
            mean_length += lng
            
            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            ts_node_i = nodes_ts[sti,:]
            ts_node_j = nodes_ts[endi,:]
            tr_node_i = nodes_tr[sti,:]
            tr_node_j = nodes_tr[endi,:]
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:].T))
            mean_mwf += np.mean(ms_rates[edge_idx,:])
            
            total_loss_storage_supply += -1*ms_rates[edge_idx,:]*CP*(tsout_edge - tsin_edge)
            total_loss_storage_return += -1*ms_rates[edge_idx,:]*CP*(trout_edge - trin_edge)
            total_weights *= weight
            total_dts_supply += np.mean(np.abs(tsin_edge - tsout_edge))
            total_dts_return += np.mean(np.abs(trin_edge - trout_edge))
            outside_dts_supply += (weight.dot(np.abs(ts_node_i - ts_node_j))) / len(weight)
            outside_dts_return += (weight.dot(np.abs(tr_node_i - tr_node_j))) / len(weight)
            
            ts_out = tsout_edge.T
            df_outputs.insert(loc=ooo, column=f'Tsout_pipe{edge_idx}->{endi+1}', value=ts_out)
            ooo += 1

            tr_in = trin_edge.T
            df_input_features.insert(loc=iii, column=f'Trin_pipe{edge_idx}->{endi+1}', value=tr_in*weight)
            iii += 1
                
            ts_in = tsin_edge.T
            df_temperatures_ins.insert(loc=lll, column=f'Tsin_pipe{edge_idx}->{endi+1}', value=ts_in)
            lll +=1

        
        # Compute smoothness normalized
        temporal_smoothness_index = []
        for t in range(x_series.shape[2]):
            x_ = x_series[:,:,t] / (np.linalg.norm(x_series[:,:,t]))
            sm_index = np.dot(x_.T, np.dot(weight_series[:,:,t],x_))
            sm_index = np.trace(sm_index) / np.linalg.norm(weight_series[:,:,t])
            temporal_smoothness_index.append(sm_index) 
        
        cluster_physical_metrics = {
            'loss_storage_supply': total_loss_storage_supply,
            'loss_storage_return': total_loss_storage_return,
            'weigths': total_weights,
            'total_dt_decrease_supply': total_dts_supply,
            'total_dt_decrease_return': total_dts_return,
            'outside_pipes_dt_ts': outside_dts_supply,
            'outside_pipes_dt_tr': outside_dts_return,
            'inside_nodes_mean_ts_difference': ts_difference_mean,
            'inside_nodes_mean_tr_difference': ts_difference_mean,
            'inside_nodes_mean_demand_evolution_difference': demands_evolution_difference_mean,
            'nb_edges': len(inner_edges) + len(in_going_edges) + len(out_going_edges),
            'normalized_smoothness_index_temporal': temporal_smoothness_index,
            'mean_diameter': mean_diameter / (len(in_going_edges) + len(out_going_edges) + len(inner_edges)),
            'mean_length': mean_length / (len(in_going_edges) + len(out_going_edges) + len(inner_edges)),
            'mean_mwf': mean_mwf / (len(in_going_edges) + len(out_going_edges) + len(inner_edges)),
        }
        return df_input_features, df_outputs, df_temperatures_ins, cluster_physical_metrics
    
    def generate_sequential_input_data_v6(self, cluster_of_nodes: list, pooling_method: 'sum'):
        
        dict_inputs = self.dict_physical_values
        inner_edges, in_going_edges, out_going_edges, _ = self.get_cluster_qualities_and_identify_connecting_pipes(cluster_of_nodes) # at time = 0
        
        demands = dict_inputs['Demands'].copy()
        ms_rates = dict_inputs['mw'].copy()
        pipes_tsin = dict_inputs['Tsin'].copy()
        pipes_tsout = dict_inputs['Tsout'].copy()
        pipes_trin = dict_inputs['Trin'].copy()
        pipes_trout = dict_inputs['Trout'].copy()
        nodes_ts = dict_inputs['Ts_node'].copy()
        nodes_tr = dict_inputs['Tr_node'].copy()
        nodes_trcs = dict_inputs['Trc'].copy()
        
        mean_diameter = 0
        mean_length = 0
        mean_mwf = 0
        
        df_input_features = pd.DataFrame()
        iii = 0
            
        df_outputs = pd.DataFrame()
        ooo = 0
        
        df_temperatures_ins = pd.DataFrame() # in going temperatures to pipes to compute loss performance
        lll = 0
        
        treated_nodes = []
        array_to_copy = np.zeros_like(np.zeros_like(pipes_trin[0,:]))
        total_loss_storage_supply = np.zeros_like(array_to_copy)
        total_loss_storage_return = np.zeros_like(array_to_copy)
        total_weights = np.ones_like(array_to_copy)
        total_dts_supply = 0
        total_dts_return = 0
        spatial_smoothness_demand = 0 # W_i,j * (D_i - D_j)^2 for all inner i,j and all time step t (sur les noeuds à aggreger)
        spatial_smoothness_ts = 0 # sur aussi les pipes adjacentes
        spatial_smoothness_tr = 0
        outside_dts_supply = 0
        outside_dts_return = 0
        
        inner_nodes_degree = {}
        aggregated_demands_per_nodes = {}
        aggregated_demands_with_temps_per_nodes = {}
        sum_trcs_per_nodes = {}
        
        nb_nodes = len(cluster_of_nodes)
        
        demands_nodes_features = np.zeros_like(demands[0,:].T)
        for el in cluster_of_nodes:
            node_index = int(el)-1
            demands_node = demands[node_index,:].T
            demands_nodes_features += demands_node
            inner_nodes_degree[node_index] = 0
            aggregated_demands_per_nodes[node_index] = demands_node
            aggregated_demands_with_temps_per_nodes[node_index] = np.zeros_like(demands[0,:].T)
            sum_trcs_per_nodes[node_index] = nodes_trcs[node_index]
            
        if pooling_method == 'sum':
            df_input_features.insert(loc=iii, column=f'Sum demand features', value=demands_nodes_features)
            iii += 1
        
        # Laplacian L = D - W
        # Normalized laplacian normL = 1 - (1/sqrt(D))W(1/sqrt(D))
        # Local smoothness = X^T.normL.X
        # Normalized local smoothness (also smoothness index [Local Smoothness of Graph Signals, M. Dakovic et al., 2019])
        # = (X^T.normL.X)/(X^T.X)
        
        # We compute for all time steps
        for edge_idx in inner_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            lambda_f = 1 + np.mean(np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:])))
            
            inner_nodes_degree[sti] += 1
            inner_nodes_degree[endi] += 1
            
            sum_trcs_per_nodes[sti] += nodes_trcs[endi]
            sum_trcs_per_nodes[endi] += nodes_trcs[sti]
            
            aggregated_demands_per_nodes[sti]  += lambda_f * demands[endi,:].T
            aggregated_demands_per_nodes[endi]  += lambda_f * demands[sti,:].T
            
            aggregated_demands_with_temps_per_nodes[sti] += lambda_f * nodes_trcs[endi] * demands[endi,:].T
            aggregated_demands_with_temps_per_nodes[endi] += lambda_f * nodes_trcs[sti] * demands[sti,:].T
        
        
        aggregated_demands_with_temps_per_nodes[sti] /= sum_trcs_per_nodes[sti]
        aggregated_demands_with_temps_per_nodes[endi] /= sum_trcs_per_nodes[endi]
        
        
        x_series = np.zeros(shape=(len(inner_nodes_degree.keys()), 4, len(demands_nodes_features))) # N x NbFeatures x TimeSteps
        weight_series = np.zeros(shape=(len(inner_nodes_degree.keys()), len(inner_nodes_degree.keys()), len(demands_nodes_features)))      
        nodes_associated_index_in_x = {}
        asso_index = 0
        for node_idx in inner_nodes_degree.keys():
            demand_node = demands[node_idx,:]
            ts_node = nodes_ts[node_idx,:]
            tr_node = nodes_tr[node_idx,:]
            ps_node = np.zeros_like(ts_node)
            x_series[asso_index,0,:] = ts_node
            x_series[asso_index,1,:] = tr_node
            x_series[asso_index,2,:] = demand_node
            x_series[asso_index,3,:] = ps_node
            nodes_associated_index_in_x[node_idx] = asso_index
            asso_index += 1
        
    
        for edge_idx in in_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            mean_diameter += d
            mean_length += lng

            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            ts_node_i = nodes_ts[sti,:]
            ts_node_j = nodes_ts[endi,:]
            tr_node_i = nodes_tr[sti,:]
            tr_node_j = nodes_tr[endi,:]
            demand_node_i = demands[sti,:]
            demand_node_j = demands[endi,:]
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:].T))
            mean_mwf += np.mean(ms_rates[edge_idx,:])
            
            total_loss_storage_supply += -1*ms_rates[edge_idx,:]*CP*(tsout_edge - tsin_edge)
            total_loss_storage_return += -1*ms_rates[edge_idx,:]*CP*(trout_edge - trin_edge)
            total_dts_supply += np.mean(np.abs(tsin_edge - tsout_edge))
            total_dts_return += np.mean(np.abs(trin_edge - trout_edge))
            outside_dts_supply += (weight.dot(np.abs(ts_node_i - ts_node_j))) / len(weight)
            outside_dts_return += (weight.dot(np.abs(tr_node_i - tr_node_j))) / len(weight)
            total_weights *= weight
            
            if pooling_method == 'weighted_sum':
                eq_demand_outgoing_remixing_mode = aggregated_demands_per_nodes[endi]
                df_input_features.insert(loc=iii, column=f'Agg. demand at node {endi+1}', value=eq_demand_outgoing_remixing_mode)
                iii +=1
            
            elif pooling_method == 'weighted_trc_sum':
                eq_demand_outgoing_remixing_mode = aggregated_demands_with_temps_per_nodes[endi]
                df_input_features.insert(loc=iii, column=f'Agg. demand at node {endi+1}', value=eq_demand_outgoing_remixing_mode)
                iii +=1
                
            ts_in = tsin_edge.T
            df_input_features.insert(loc=iii, column=f'Tsin_pipe{edge_idx}<-{sti+1}', value=ts_in*weight)
            iii +=1
            
            tr_out = trout_edge.T  
            df_outputs.insert(loc=ooo, column=f'Trout_pipe{edge_idx}->{sti+1}', value=tr_out)
            ooo += 1
            
            tr_in = trin_edge.T
            df_temperatures_ins.insert(loc=lll, column=f'Trin_pipe{edge_idx}<-{sti+1}', value=tr_in)
            lll +=1
            
            # Smoothness
            # spatial_smoothness_tr += weight*(trout_edge - trin_edge)
            # spatial_smoothness_ts += weight*(tsout_edge - tsin_edge)
            
        demands_evolution_difference_mean = 0
        ts_difference_mean = 0
        tr_difference_mean = 0
        for edge_idx in inner_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            mean_diameter += d
            mean_length += lng
            
            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            ts_node_i = nodes_ts[sti,:]
            ts_node_j = nodes_ts[endi,:]
            tr_node_i = nodes_tr[sti,:]
            tr_node_j = nodes_tr[endi,:]
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:]))
            mean_mwf += np.mean(ms_rates[edge_idx,:])
            
            total_loss_storage_supply += -1*ms_rates[edge_idx,:]*CP*(tsout_edge - tsin_edge)
            total_loss_storage_return += -1*ms_rates[edge_idx,:]*CP*(trout_edge - trin_edge)
            total_weights *= weight
            total_dts_supply += np.mean(np.abs(tsin_edge - tsout_edge))
            total_dts_return += np.mean(np.abs(trin_edge - trout_edge))
            
            # Smoothness
            demand_sti = self.get_weighted_difference_signal(demands[sti,:])
            demand_endi = self.get_weighted_difference_signal(demands[endi,:])
            demands_evolution_difference_mean += np.mean(np.abs(demand_sti - demand_endi))
            ts_difference_mean += np.mean(np.abs(ts_node_j - ts_node_i))
            tr_difference_mean += np.mean(np.abs(tr_node_j - tr_node_i))
            
            d_i = inner_nodes_degree[sti]
            d_j = inner_nodes_degree[endi]
            weight_series[nodes_associated_index_in_x[sti], nodes_associated_index_in_x[sti], :] = np.ones_like(weight)
            weight_series[nodes_associated_index_in_x[endi], nodes_associated_index_in_x[endi], :] = np.ones_like(weight)
            weight_series[nodes_associated_index_in_x[sti], nodes_associated_index_in_x[endi], :] = weight / (d_i*d_j)
            weight_series[nodes_associated_index_in_x[endi], nodes_associated_index_in_x[sti], :] = weight / (d_i*d_j)
        
        demands_evolution_difference_mean /= len(cluster_of_nodes)
        tr_difference_mean /= len(cluster_of_nodes)
        ts_difference_mean /= len(cluster_of_nodes)
          
        for edge_idx in out_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edge_features.iloc[edge_idx]
            h = row['h']
            lng = row['length'] * 1e3
            d = row['Diameter']
            
            mean_diameter += d
            mean_length += lng
            
            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            ts_node_i = nodes_ts[sti,:]
            ts_node_j = nodes_ts[endi,:]
            tr_node_i = nodes_tr[sti,:]
            tr_node_j = nodes_tr[endi,:]
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:].T))
            mean_mwf += np.mean(ms_rates[edge_idx,:])
            
            total_loss_storage_supply += -1*ms_rates[edge_idx,:]*CP*(tsout_edge - tsin_edge)
            total_loss_storage_return += -1*ms_rates[edge_idx,:]*CP*(trout_edge - trin_edge)
            total_weights *= weight
            total_dts_supply += np.mean(np.abs(tsin_edge - tsout_edge))
            total_dts_return += np.mean(np.abs(trin_edge - trout_edge))
            outside_dts_supply += (weight.dot(np.abs(ts_node_i - ts_node_j))) / len(weight)
            outside_dts_return += (weight.dot(np.abs(tr_node_i - tr_node_j))) / len(weight)
            
            if pooling_method == 'weighted_sum':
                eq_demand_outgoing_remixing_mode = aggregated_demands_per_nodes[sti]
                df_input_features.insert(loc=iii, column=f'Agg. demand at node {sti+1}', value=eq_demand_outgoing_remixing_mode)
                iii +=1
            
            elif pooling_method == 'weighted_trc_sum':
                eq_demand_outgoing_remixing_mode = aggregated_demands_with_temps_per_nodes[sti]
                df_input_features.insert(loc=iii, column=f'Agg. demand at node {sti+1}', value=eq_demand_outgoing_remixing_mode)
                iii +=1
            
            ts_out = tsout_edge.T
            df_outputs.insert(loc=ooo, column=f'Tsout_pipe{edge_idx}->{endi+1}', value=ts_out)
            ooo += 1

            tr_in = trin_edge.T
            df_input_features.insert(loc=iii, column=f'Trin_pipe{edge_idx}<-{endi+1}', value=tr_in*weight)
            iii += 1
                
            ts_in = tsin_edge.T
            df_temperatures_ins.insert(loc=lll, column=f'Tsin_pipe{edge_idx}<-{endi+1}', value=ts_in)
            lll +=1

        
        # Compute smoothness normalized
        temporal_smoothness_index = []
        for t in range(x_series.shape[2]):
            x_ = x_series[:,:,t] / (np.linalg.norm(x_series[:,:,t]))
            sm_index = np.dot(x_.T, np.dot(weight_series[:,:,t],x_))
            sm_index = np.trace(sm_index) / np.linalg.norm(weight_series[:,:,t])
            temporal_smoothness_index.append(sm_index) 
        
        cluster_physical_metrics = {
            'loss_storage_supply': total_loss_storage_supply,
            'loss_storage_return': total_loss_storage_return,
            'weigths': total_weights,
            'total_dt_decrease_supply': total_dts_supply,
            'total_dt_decrease_return': total_dts_return,
            'outside_pipes_dt_ts': outside_dts_supply,
            'outside_pipes_dt_tr': outside_dts_return,
            'inside_nodes_mean_ts_difference': ts_difference_mean,
            'inside_nodes_mean_tr_difference': ts_difference_mean,
            'inside_nodes_mean_demand_evolution_difference': demands_evolution_difference_mean,
            'nb_edges': len(inner_edges) + len(in_going_edges) + len(out_going_edges),
            'normalized_smoothness_index_temporal': temporal_smoothness_index,
            'mean_diameter': mean_diameter / (len(in_going_edges) + len(out_going_edges) + len(inner_edges)),
            'mean_length': mean_length / (len(in_going_edges) + len(out_going_edges) + len(inner_edges)),
            'mean_mwf': mean_mwf / (len(in_going_edges) + len(out_going_edges) + len(inner_edges)),
        }
        return df_input_features, df_outputs, df_temperatures_ins, cluster_physical_metrics
    
    def get_cluster_qualities_and_identify_connecting_pipes(self, nodes_in_cluster: list):
        """Computes the qualities of clusters for undirected and the connecting pipes

        Args:
            nodes_in_cluster (list): Cluster of nodes
            
        Returns:
            tuple: inner edges, in going edges, out going edges, qualities  [density, total inner degree, total external degree, cluster volume, cut size, cut ratio, conductance, max diameter]
        """
        internal_edges_idx = []
        external_edges_idx_pointing_in = []
        external_edges_idx_pointing_out = []
        inner_pipes_total_length = [] # Longueur totale du pipe
        
        pipes = {}
        nodes_internal_indegree = {}
        nodes_external_indegree = {}
        nodes_internal_outdegree = {}
        nodes_external_outdegree = {}
        
        for node in nodes_in_cluster:
            nodes_internal_indegree[node]=0
            nodes_external_indegree[node]=0
            nodes_internal_outdegree[node]=0
            nodes_external_outdegree[node]=0
        
        # on cherche dans la dict
        for edge_idx in self.edges_nodes:
            (st_n, en_n) = self.edges_nodes[edge_idx]
            if (st_n+1) in nodes_in_cluster:
                
                if (en_n+1) in nodes_in_cluster:
                    # ca veut dire que c'est une connexion interne
                    internal_edges_idx.append(edge_idx)
                    nodes_internal_outdegree[st_n+1] += 1  
                    nodes_internal_indegree[en_n+1] += 1
                else:
                    # ca veut dire que c'est une connexion externe
                    external_edges_idx_pointing_out.append(edge_idx)
                    nodes_external_outdegree[st_n+1] += 1
            
            elif (en_n+1) in nodes_in_cluster and (st_n+1) not in nodes_in_cluster:
                external_edges_idx_pointing_in.append(edge_idx)
                nodes_external_indegree[en_n+1] += 1
                
        max_diameter = 0
        for i in range(len(nodes_in_cluster)):
            node_i_indx = nodes_in_cluster[i] - 1
            for j in range(i+1, len(nodes_in_cluster)):
                node_j_indx = nodes_in_cluster[j] - 1
                d_i_j = np.linalg.norm((np.array(self.nodes_coordinates[node_i_indx]) - np.array(self.nodes_coordinates[node_j_indx])))
                max_diameter = max(max_diameter, d_i_j)
          
        # Pour eviter les repetitions on prend un set, similaire à HashSet pour C#
        inner_edges = set(internal_edges_idx)
        in_going_edges = set(external_edges_idx_pointing_in)
        out_going_edges = set(external_edges_idx_pointing_out)
        
        n_e = len(inner_edges)
        n_c = len(nodes_in_cluster)
        
        if n_c == 1:
            inner_density = 0
        else:
            inner_density = n_e / (n_c * (n_c -1)) # Density for directed graphs
            
        inner_degree = n_e
        scaled_density = n_c * inner_density # (Vinh-Loc Dao et al., 2018)
        external_degree = len(in_going_edges) + len(out_going_edges)
        
        # https://www.cis.upenn.edu/~cis5150/cis515-15-spectral-clust-chap5.pdf
        # https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf
        # Volume set of nodes = Sum vol(node) wher vol(node) = total degree
        # Volume of a cut = total out degree of cut = cut size
        # Cut ratio = cut size / n_c where cut size = sum of external connections
        # Normalized cut = cut size * (1/vol(set S) + 1/vol(set S_bar))
        # Conductance = volume cut (S) / min(vol set S, vol set S_bar) in our case vol set S < (always) vol set S_bar
        
        # Basic qualities
        total_cluster_degree = inner_degree + external_degree
        # Interesting qualities
        cut_size = external_degree
        cut_ratio = cut_size / n_c
        conductance = cut_size / total_cluster_degree
        
        nodes_outdegree_fraction = []
        nodes_indegree_fraction = [] # in from external edges
        for node in nodes_external_outdegree:
            total_outdegree = nodes_external_outdegree[node]+nodes_internal_outdegree[node]
            if total_outdegree == 0:
                nodes_outdegree_fraction.append(0.0) # Peut arriver pour les noeuds de bout de réseaux
            else:
                nodes_outdegree_fraction.append(nodes_external_outdegree[node]/(nodes_external_outdegree[node]+nodes_internal_outdegree[node]))
        for node in nodes_external_indegree:
            total_indegree = nodes_external_indegree[node]+nodes_internal_indegree[node]
            if total_indegree == 0:
                nodes_indegree_fraction.append(0.0) # Normalement que pour les sources
            else:
                nodes_indegree_fraction.append(nodes_external_indegree[node]/(nodes_external_indegree[node]+nodes_internal_indegree[node]))
        
        average_outdegree_fraction = np.mean(np.array(nodes_outdegree_fraction))
        max_outdegree_fraction = np.max(np.array(nodes_outdegree_fraction))
        average_indegree_fraction = np.mean(np.array(nodes_indegree_fraction))
        max_indegree_fraction = np.max(np.array(nodes_indegree_fraction))
        
        # print(nodes_internal_indegree.values())
        # Qualities = [density, total inner degree, total external degree, cluster volume, cut size, cut ratio, conductance, cut value, max diameter]
        qualities = {
            # Internal degree and connectivity
            'density': inner_density,
            'scaled_density': scaled_density,
            'nodes_mean_internal_indegree': sum(nodes_internal_indegree.values()) / len(nodes_internal_indegree),
            'nodes_max_internal_indegree': max(nodes_internal_indegree.values()),
            'nodes_mean_internal_outdegree': sum(nodes_internal_outdegree.values()) / len(nodes_internal_outdegree),
            'nodes_max_internal_outdegree': max(nodes_internal_outdegree.values()),
            
            # External degree and connectivity
            'average_outdegree_fraction': average_outdegree_fraction,
            'max_outdegree_fraction': max_outdegree_fraction,
            'average_indegree_fraction': average_indegree_fraction,
            'max_indegree_fraction': max_indegree_fraction,
            'cut_size': cut_size,
            'cut_ratio': cut_ratio,
            'conductance': conductance,
            
            # Surface
            'max_diameter': max_diameter,
        }
        return inner_edges, in_going_edges, out_going_edges, qualities