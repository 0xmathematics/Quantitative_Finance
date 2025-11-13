#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 13:25:43 2025

@author: Hang Miao
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import cos,sin,pi,exp,log,log2,log10,sqrt,ceil,floor


class binomial_interest_rate:
    def __init__(self, par_rates=None, spot_rates=None):
        if (par_rates is None) == (spot_rates is None):
            raise ValueError("You must provide exactly one of par_rates or spot_rates.")
        
        if par_rates is not None:
            self.par_rates = par_rates
            self.spot_rates = self._convert_par_to_spot(par_rates)
        else:
            self.spot_rates = spot_rates
        #print(self.spot_rates)    
        self.forward_rates = self._convert_spot_to_forward(self.spot_rates ) 
        self.t_level = len(self.spot_rates)
        self.tree = self._build_binomial_tree(self.t_level-1)
        self.tree_pos = self._lattice_positions(self.tree)
        self.rates_array =  {}   
        self.rates_array[0] = [self.spot_rates[0]]
        

        
        
        #self.node_value_labels = {n: f'{n} \n {self.tree.nodes[n]["forward_rate"]:.2f}' for n in self.tree.nodes}
        
    def _convert_par_to_spot(self, par_rates_, PV = 100, FV=100):
        par_rates = np.array(par_rates_)
        N = par_rates.shape[0]
        spot_rates = []
        for i, par_rate in enumerate(par_rates):
            #i = i+1
            pmt = FV*par_rate
            
            if i == 0:
                spot_rate = ((FV+pmt)/PV-1 )
                spot_rates.append(spot_rate)
                #print(spot_rate)
                #print( par_rates[0])
                continue
            else:
                sum_PV = 0
                for j in range(i):
                    sum_PV = sum_PV+ pmt/(1+spot_rates[j])**(j+1)
                spot_rate = ((FV+pmt)/(PV-sum_PV))**(1/(i+1)) -1
                spot_rates.append(spot_rate)
        #print(i)
        return np.array(spot_rates)
    def _convert_spot_to_par(self, spot_rates):
        # Placeholder conversion logic
        print("Converting spot rates to par rates...")
        return [r / 0.95 for r in spot_rates]  # Example

    
    def _convert_spot_to_forward(self, spot_rates_ = None):

        if (self.spot_rates is None) and (spot_rates_ is None):
            raise ValueError("Spot to Forward rate error: You must provide exactly one of par_rates or spot_rates.")
            
        spot_rates = spot_rates_ if spot_rates_ is not None else self.spot_rates
        N = spot_rates.shape[0]
        
        forward_rates = []
        #for i, spot_rate in enumerate(spot_rates):
        for i in range( spot_rates.shape[0]-1 ):
            
            
            #print(i)
            #print(spot_rates[i])
            z_current = spot_rates[i]
            z_next = spot_rates[i+1]
            i = i+1
            #print(z_current)
            # (1 + z_i)**i * (1+f_i_1) = (1+z_{i+1})**(i+1)
            f_i_1 = (1+z_next)**(i+1) / (1 + z_current)**(i) -1
    
            forward_rates.append(f_i_1)
    
        return forward_rates
    
    def PV_binomial_interest_tree_bond(self, coupoun_rate, maturity, FV= 100):
        pmt = FV*coupoun_rate
        t_level = maturity-1
        
        #par_rate = par_rates[t_level] 
        terminal_price = (coupoun_rate+1)*FV
        
        
        forward_rates_initial = np.array( self.rates_array[t_level] )
        PV_array = {}
        
        PV_list_cur = terminal_price/( forward_rates_initial + 1 )  
        PV_array[t_level] = PV_list_cur
        for t in np.arange(t_level,0,-1):
            #print(t)
            PV_list = PV_list_cur.copy()
            PV_list_cur = []
            #node_pre_price = []
            for i in range(len(PV_list)-1):
                PV = ( 0.5*PV_list[i] + 0.5*PV_list[i+1]  +  coupoun_rate*100 )/( 1 +self.rates_array[t-1][i])
                PV_list_cur.append(PV)
            
            PV_array[t-1] = PV_list_cur
        return PV_array
        
        
        
        
        
    def calibration_binomial_interest_rate(self, sigma = 0.15):
        
        T = len( self.par_rates )
        for t_level in range(1,T):

            forward_rate_l = self.forward_rates[t_level-1]*exp(-t_level*sigma)
            
            # Run optimization starting from an initial guess (say x=0)
            result = minimize(lambda x: self.objective(x, t_level, sigma), x0=forward_rate_l)  #0.0125 0.012074059789530169
            
            print("Optimal x:", result.x[0])
            print("Function value:", self.PV_binomial_interest_tree(result.x[0], t_level, sigma)[0][0] )
            
            forward_rates_final = [result.x[0]*exp(2*sigma*i) for i in range(t_level+1) ]
            
            self.rates_array[t_level] = forward_rates_final
            
            
        for node in self.tree.nodes:
            t, i = node
            self.tree.nodes[node]["forward_rate"]= self.rates_array[t][i] 
 
        return self.rates_array
    
    def PV_binomial_interest_tree(self,  forward_rate_l, t_level=1, sigma = 0.15, par_rates_ = None ):
        par_rates = par_rates_ if par_rates_ is not None else self.par_rates
        
        par_rate = par_rates[t_level] 
        terminal_price = (par_rate+1)*100
        forward_rates_initial = np.array([forward_rate_l*exp(2*sigma*i) for i in range(t_level+1) ] ) 
        PV_array = {}
        
        PV_list_cur = terminal_price/( forward_rates_initial + 1 )  
        PV_array[t_level] = PV_list_cur
        for t in np.arange(t_level,0,-1):
            #print(t)
            PV_list = PV_list_cur.copy()
            PV_list_cur = []
            #node_pre_price = []
            for i in range(len(PV_list)-1):
                PV = ( 0.5*PV_list[i] + 0.5*PV_list[i+1]  +  par_rate*100 )/( 1 +self.rates_array[t-1][i])
                PV_list_cur.append(PV)
            
            PV_array[t-1] = PV_list_cur
        
        return PV_array
        

    # Define the objective (distance from 100)
    def objective(self, x, t_level, sigma):  
        PV_array = self.PV_binomial_interest_tree(x, t_level, sigma  )  
        discrepency =    abs( PV_array[0][0]  - 100)
        return discrepency
    
    
    def _build_binomial_tree(self, T_: int = None):
        """
        Build a directed recombining (binomial) tree with T time steps.
        Nodes are (t, i) with t=0..T and i=0..t.
        """
        T = T_ if T_ is not None else (self.t_level-1)
        G = nx.DiGraph()
        # add nodes
        for t in range(T + 1):
            for i in range(t + 1):
                G.add_node((t, i), t=t, i=i)
        # add edges (down and up)
        for t in range(T):
            for i in range(t + 1):
                G.add_edge((t, i), (t + 1, i), move="down")     # same i
                G.add_edge((t, i), (t + 1, i + 1), move="up")   # i+1
                
        # initialize node label        
        G.nodes[(0, 0)]["forward_rate"] = self.spot_rates[0]
        # initialization assign nan to all other nodes
        for node in G.nodes:
            if "forward_rate" not in G.nodes[node]:
                G.nodes[node]["forward_rate"] = float("nan")        

        #node_value_labels = {n: f'{n} \n {G.nodes[n]["forward_rate"]:.2f}' for n in G.nodes}
                
        return G
    
    def _lattice_positions(self, G_ = None, x_gap=1.5, y_gap=1.0):
        """
        Place nodes on a grid: time t on x-axis, state i on y-axis,
        centered vertically per time step.
        """
        G = G_ if G_ is not None else self.tree
        # collect max time
        T = max(t for (t, i) in G.nodes)
        pos = {}
        for t in range(T + 1):
            level_nodes = [(t, i) for (t, i) in G.nodes if t == t]  # just iterate i
            n_level = t + 1
            # center states around 0 for symmetry
            y0 = -(n_level - 1) / 2.0
            for i in range(n_level):
                pos[(t, i)] = (t * x_gap, (y0 + i) * y_gap)
        return pos

    def plot_binomial_tree(self, G = None, pos = None):
        if G == None or pos == None:
            G = self.tree; pos = self.tree_pos
            #pos = self._lattice_positions(G)
            
        plt.figure(figsize=(10, 5))
        nx.draw(
            G, pos,
            with_labels=False,
            node_size=400,
            node_color="#ddddff",
            arrows=False,   # tree direction is obvious; set True if you want arrows
            linewidths=0.5,
            width=1.0
        )
        
        # label nodes as (t,i)
        #node_labels = {n: f"{n}" for n in G.nodes}
        #nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        
        
        node_value_labels = {n: f'{n} \n {G.nodes[n]["forward_rate"]:.4%}' for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_value_labels, font_size=10)
        # (optional) color edges by move type
        edge_colors = ["#2ca02c" if G.edges[e]["move"] == "up" else "#1f77b4" for e in G.edges]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.2)
        
        plt.axis("off")
        plt.tight_layout()
        plt.show() 
