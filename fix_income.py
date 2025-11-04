# -*- coding: utf-8 -*-
"""
Created on Tue July 29 11:25:26 2025
@author: Hang Miao
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import cos,sin,pi,exp,log,log2,log10,sqrt,ceil,floor

#%% 
# fixed income book
# p25 example 8
DF1 = 0.9524
DF2 = 0.89
DF3 = 0.8163
DF4 = 0.735

array_DF = np.array( [0.9524, 0.89, 0.8163, 0.735 ])

# array_DF* r_fix + 1*DF4 = 1
r_fix =  (1-1*DF4)/ array_DF.sum()

# p49
z1 = 4  *0.01
z2 = 5  *0.01
z3 = 6  *0.01

# (1+z1)*(1+f_1_1) = (1+z2)**2
f_1_1 = (1+z2)**2/(1+z1)-1

# (1+z1)*(1+f_1_2)**2 = (1+z3)**3
f_1_2 = sqrt((1+z3)**3/(1+z1))-1

# q9
f_2_1 = (1+z3)**3/(1+z2)**2-1

# q10
F_3_2 = 0.8479

# 1/(1+f_3_2) = F_3_2
f_3_2 = 1/F_3_2-1

#  (1+z5)**5 = (1+z3)**3 * (1+f_3_2)**2
#  DF_3 * F_3_2 = DF_5
DF_5 =  1/(1+z3)**3 * F_3_2

# q11
z1 = 4  *0.01
f_1_1 = 6  *0.01
f_2_1 = 8  *0.01

# (1+z1)*(1+f_1_1)*(1+f_2_1) = (1+z3)**3
z3 = ( (1+z1)*(1+f_1_1)*(1+f_2_1) )**(1/3)-1

# q12
z1 = 5  *0.01
F_1_1 = 0.9346

# (1+z1) * (1/F_1_1) = (1+z2)**2
DF_2 = F_1_1/(1+z1)

# p51  Q 18-24

# Par rate vs Spot rate
pz2 = 2.99  * 0.01
pz3 = 3.48  * 0.01
pz4 = 3.95  * 0.01
pz5 = 4.37  * 0.01

z1 = 2.5 * 0.01
z2 = 3 * 0.01
z3 = 3.5 * 0.01
z4 = 4 * 0.01
#z5 = ?
# 1 + pz1 = 1 + z1
# pz2
PV = 1/(1+z2)**2
# 
2.99/(1+z1) + 102.99/(1+z2)**2
3.48/(1+z1) + 3.48/(1+z2)**2 + 103.48/(1+z3)**3
3.95/(1+z1) + 3.95/(1+z2)**2 + 3.95/(1+z3)**3 + 103.95/(1+z4)**4
# 4.37/(1+z1) + 4.37/(1+z2)**2 + 4.37/(1+z3)**3 + 4.37/(1+z4)**4 +  104.37/(1+z5)**5 = 100
z5 = (104.37/ ( 100 - ( 4.37/(1+z1) + 4.37/(1+z2)**2 + 4.37/(1+z3)**3 + 4.37/(1+z4)**4 ) ))**(1/5)-1


#  Q20 

f_3_1 = (1+z4)**4/(1+z3)**3-1


# Q22
PV = 6/(1+z1) + 6/(1+z2)**2 + 6/(1+z3)**3 + 6/(1+z4)**4 +  106/(1+z5)**5 



# Q25-29
# Q25
z1 = 2.25 * 0.01
z2 = 2.7 * 0.01
z3 = 3.3 * 0.01
z4 = 4.05 * 0.01

# (1+z1)*(1+f_1_1) = (1+z2)**2
f_1_1 =  (1+z2)**2/(1+z1) -1
F_1 = 1/(1+f_1_1)

# Q27
PV = 1/(1+z4+0.7*0.01)**4
FV = 1/(1+z2+0.3*0.01)**2

# PV*(1+YTM2)**2 = FV
sqrt(FV/PV)-1

# Q28
pmt = (4.15)*0.01
z_spread = 65 * 0.0001
pmt/(1+z1+z_spread) + (1+pmt)/(1+z2+z_spread)**2

# Q37
z1 = 14 * 0.01
z2 = 12.4 * 0.01
z3 = 11.8 * 0.01
z4 = 11.00 * 0.01
z5 = 10.7 * 0.01

# (1+z3)**3*(1+f_3_2)**2 = (1+z5)**5
f_3_2 = ( (1+z5)**5/(1+z3)**3 )**(1/2)-1
#%% 
# section arbitrage-free valuation framework

# P73 Example 2
YTM1 = 2 *0.01
YTM2 = 3 *0.01
YTM3 = 4 *0.01


PV = 102.7751
5/(1+YTM3) + 5/(1+YTM3)**2 + 105/(1+YTM3)**3 

z1 = YTM


#%% 

# bootstrap method
#  converting par rates to spot rates

par_rates = [0.01, 0.012, 0.0125, 0.014, 0.018]

def par_spot(par_rates_, PV = 100, FV=100):
    
    par_rates = np.array(par_rates_)
    N = par_rates.shape[0]
    
    spot_rates = []
    for i, par_rate in enumerate(par_rates):
        #i = i+1
        
        print(i)
        print(par_rate)
        
        pmt = FV*par_rate
        
        if i == 0:
            spot_rate = ((FV+pmt)/PV-1 )
            spot_rates.append(spot_rate)
            print(spot_rate)
            print( par_rates[0])
            continue
        else:
            sum_PV = 0
            for j in range(i):
                sum_PV = sum_PV+ pmt/(1+spot_rates[j])**(j+1)
            spot_rate = ((FV+pmt)/(PV-sum_PV))**(1/(i+1)) -1
            spot_rates.append(spot_rate)
    print(i)
    return spot_rates
spot_rates = par_spot(par_rates )

#%% 

# derive binomial tree model 
#  converting spot rates to one year forward rates

par_rates = [0.01, 0.012, 0.0125, 0.014, 0.018]

def spot_forward(spot_rates_):
    
    spot_rates = np.array(spot_rates_)
    N = spot_rates.shape[0]
    
    forward_rates = []
    #for i, spot_rate in enumerate(spot_rates):
    for i in range( spot_rates.shape[0]-1 ):
        
        
        print(i)
        #print(spot_rates[i])
        z_current = spot_rates[i]
        z_next = spot_rates[i+1]
        i = i+1
        #print(z_current)
        # (1 + z_i)**i * (1+f_i_1) = (1+z_{i+1})**(i+1)
        f_i_1 = (1+z_next)**(i+1) / (1 + z_current)**(i) -1

        forward_rates.append(f_i_1)

    return forward_rates



spot_rates = par_spot(par_rates )
forward_rates = spot_forward(spot_rates )



#%% 
# calibrating the binomial interest rate tree 
par_rates = [0.01, 0.012, 0.0125, 0.014, 0.018]
spot_rates = par_spot(par_rates )
forward_rates = spot_forward(spot_rates )


sigma = 0.15

forward_rate = forward_rates[0] # 0.014028056112224796
par_rate = par_rates[1] # 0.012

forward_rate_root = 0.01
forward_rate_l = 0.0125 #forward_rates[0]*exp(-sigma)
forward_rate_h = forward_rate_l*exp(2*sigma)

forward_rate_root = 0.01
forward_rate_l = 0.0125 #forward_rates[0]*exp(-sigma)
forward_rate_h = forward_rate_l*exp(2*sigma)

terminal_price  = (par_rate+1)*100
root_price = (terminal_price/(1+forward_rate_h)*0.5 + terminal_price/(1+forward_rate_l)*0.5 + par_rate*100)/(1+forward_rate_root)


from scipy.optimize import minimize

# Define your function
def PV_interest_rate_binominal(forward_rate_l,sigma = 0.15):
    par_rate = 0.012
    forward_rate_root = 0.01
    #forward_rate_l = 0.0125 #forward_rates[0]*exp(-sigma)
    forward_rate_h = forward_rate_l*exp(2*sigma)
    
    terminal_price  = (par_rate+1)*100
    root_price = (terminal_price/(1+forward_rate_h)*0.5 + terminal_price/(1+forward_rate_l)*0.5 + par_rate*100)/(1+forward_rate_root)

    return root_price

# Define the objective (distance from 100)
def objective(x):
    return abs(PV_interest_rate_binominal(x[0]) - 100)

# Run optimization starting from an initial guess (say x=0)
result = minimize(objective, x0=[0.012074059789530169])  #0.0125 0.012074059789530169

print("Optimal x:", result.x[0])
print("Function value:", PV_interest_rate_binominal(result.x[0]))




#%% 


# Define your function
def PV_interest_rate_binominal(forward_rate_l,t_level, sigma = 0.15):
    par_rate = par_rates[] 
    forward_rate_root = 0.01
    #forward_rate_l = 0.0125 #forward_rates[0]*exp(-sigma)
    forward_rate_h = forward_rate_l*exp(2*sigma)
    
    terminal_price  = (par_rate+1)*100
    root_price = (terminal_price/(1+forward_rate_h)*0.5 + terminal_price/(1+forward_rate_l)*0.5 + par_rate*100)/(1+forward_rate_root)

    return root_price

# Define the objective (distance from 100)
def objective(x):
    return abs(PV_interest_rate_binominal(x[0]) - 100)

# Run optimization starting from an initial guess (say x=0)
result = minimize(objective, x0=[0.012074059789530169])  #0.0125 0.012074059789530169

print("Optimal x:", result.x[0])
print("Function value:", PV_interest_rate_binominal(result.x[0]))


    rates_array = []
    T = len( par_rates )
    spot_rates = par_spot(par_rates )
    forward_rates = spot_forward(spot_rates )
    rates_array.append([spot_rates[0]])

rates_array[0][0]

def calibration_binomial_interest_rate(par_rates,sigma = 0.15 ):
    rates_array = []
    T = len( par_rates )
    spot_rates = par_spot(par_rates )
    forward_rates = spot_forward(spot_rates )
    rates_array.append([spot_rates[0]])
    # build binomial interest tree
    forward_tree = build_binomial_tree(T-1)
    # initialization assign nan to all other nodes
    forward_tree.nodes[(0, 0)]["forward_rate"] = par_rates[0]
    for node in forward_tree.nodes:
        if "forward_rate" not in forward_tree.nodes[node]:
            forward_tree.nodes[node]["forward_rate"] = float("nan")
            
    plot_binomial_tree(forward_tree)        
    for t in range(1,T):
        print(t)
        print(par_rates[t])
        
        forward_rate_l = forward_rates[t-1]*exp(-t*sigma)
        
        
        forward_rate_initial = [forward_rate_l*exp(2*sigma*i) for i in range(t+1) ] 
        
        
        
        
        
        
    return forward_tree 

# Define your function
def PV_interest_rate_binominal(forward_rate_l,t_level, sigma = 0.15):
    par_rate = par_rates[] 
    forward_rate_root = 0.01
    
    
    forward_rate_initial = [forward_rate_l*exp(2*sigma*i) for i in range(t+1) ] 
    #forward_rate_l = 0.0125 #forward_rates[0]*exp(-sigma)
    forward_rate_h = forward_rate_l*exp(2*sigma)
    
    terminal_price  = (par_rate+1)*100
    root_price = (terminal_price/(1+forward_rate_h)*0.5 + terminal_price/(1+forward_rate_l)*0.5 + par_rate*100)/(1+forward_rate_root)

    return root_price

        
forward_tree = calibration_binomial_interest_rate(par_rates,sigma = 0.15 )    
        
       
list(forward_tree )

forward_tree.nodes
t_level = 2
level_nodes = [(t, i) for (t, i) in forward_tree.nodes if t == t_level] 
level_nodes

node_pre = list( Forward_tree.predecessors((2,2)) )[0]

#%% 
def build_binomial_tree(T: int):
    """
    Build a directed recombining (binomial) tree with T time steps.
    Nodes are (t, i) with t=0..T and i=0..t.
    """
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
    return G


def lattice_positions(G, x_gap=1.5, y_gap=1.0):
    """
    Place nodes on a grid: time t on x-axis, state i on y-axis,
    centered vertically per time step.
    """
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


def plot_binomial_tree(G):
    
    pos = lattice_positions(G)

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
    
    
    node_value_labels = {n: f'{n} \n {G.nodes[n]["forward_rate"]:.2f}' for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_value_labels, font_size=10)
    # (optional) color edges by move type
    edge_colors = ["#2ca02c" if G.edges[e]["move"] == "up" else "#1f77b4" for e in G.edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.2)
    
    plt.axis("off")
    plt.tight_layout()
    plt.show()    
#%% 

T = len(par_rates)
Forward_tree = build_binomial_tree(T-1)
Forward_tree.edges()
Forward_tree.nodes()

Forward_tree.nodes[(0, 0)]["forward_rate"] = par_rates[0]
# initialization assign nan to all other nodes
for node in Forward_tree.nodes:
    if "forward_rate" not in Forward_tree.nodes[node]:
        Forward_tree.nodes[node]["forward_rate"] = float("nan")


def calibration_binomial_interest_rate(par_rates,sigma = 0.15 ):
    rates = []
    N = len( par_rates )
    spot_rates = par_spot(par_rates )
    forward_rates = spot_forward(spot_rates )
    forward_tree = []
    
    
    for t in range(  ):
        
        
plot_binomial_tree(Forward_tree)

Forward_tree
for t in range(1,T):
    print(t)
    print(par_rates[t])

node_pre = list( Forward_tree.predecessors((2,2)) )[0]


#%% 
import math
import matplotlib.pyplot as plt
import networkx as nx

def draw_binomial_tree(n_steps=5, S0=100.0, u=1.1, d=0.9, p=0.5,
                       annotate="price", figsize=(9, 5),
                       node_size=600, font_size=9):
    """
    Draw a recombining binomial tree using networkx.

    Parameters
    ----------
    n_steps : int
        Number of time steps (levels).
    S0 : float
        Initial asset price.
    u, d : float
        Up and down multipliers per step.
    p : float
        Risk-neutral (or chosen) up probability (only used for display).
    annotate : {"price","index","none"}
        What to show inside nodes: price value, (t,i) index, or nothing.
    figsize : tuple
        Matplotlib figure size.
    node_size : int
        Size of nodes.
    font_size : int
        Label font size.
    """
    # Build graph
    G = nx.DiGraph()

    # Create nodes: one per (t, i) where t in [0..n_steps], i in [0..t]
    # We'll store price and probability-of-path (optional) as attributes
    for t in range(n_steps + 1):
        for i in range(t + 1):
            S = S0 * (u ** i) * (d ** (t - i))
            # For display only: binomial probability of being at (t,i)
            prob = math.comb(t, i) * (p ** i) * ((1 - p) ** (t - i)) if t > 0 else 1.0
            G.add_node((t, i), S=S, prob=prob)

    # Add edges: from (t,i) -> (t+1,i) [down] and (t+1,i+1) [up]
    for t in range(n_steps):
        for i in range(t + 1):
            G.add_edge((t, i), (t + 1, i), move="d")
            G.add_edge((t, i), (t + 1, i + 1), move="u")

    # Compute positions: x = t, y = centered level
    # y spacing: put i from top to bottom but centered around 0
    pos = {}
    for t in range(n_steps + 1):
        # center the column around 0
        offset = -t / 2.0
        for i in range(t + 1):
            pos[(t, i)] = (t, offset + i)

    # Prepare node labels
    if annotate == "price":
        node_labels = {n: f"{G.nodes[n]['S']:.2f}" for n in G.nodes}
    elif annotate == "index":
        node_labels = {n: f"{n}" for n in G.nodes}
    else:  # "none"
        node_labels = {n: "" for n in G.nodes}

    # Edge labels ('u' / 'd'), optional
    edge_labels = {(u, v): G.edges[(u, v)]["move"] for (u, v) in G.edges}

    # Draw
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="#DCE7FF", edgecolors="#3B82F6", linewidths=1.5)
    nx.draw_networkx_edges(G, pos, arrows=False, width=1.0, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size)

    # Draw edge labels small and slightly transparent
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size-2, alpha=0.7,
                                 label_pos=0.35, bbox=dict(alpha=0.0, edgecolor="none"))

    plt.axis("off")
    plt.title(f"Recombining Binomial Tree (n={n_steps}, S0={S0}, u={u}, d={d}, p={p})")
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    draw_binomial_tree(n_steps=6, S0=100, u=1.1, d=0.9, p=0.5, annotate="price")










#%% 

import QuantLib as ql

# market inputs
today = ql.Date(26, 10, 2025)
ql.Settings.instance().evaluationDate = today
tenors = [ql.Period(n, ql.Years) for n in range(1,5)]
zeros  = [0.01, 0.012, 0.037, 0.038, 0.039]  # example spot rates
curve  = ql.ZeroCurve([today + t for t in tenors], zeros, ql.Actual365Fixed())
ts     = ql.YieldTermStructureHandle(curve)




#%% 
# spot curve flat, spot rate = coupon rate, then par value = market value?
z1 = 8  *0.01
z2 = 8  *0.01
z3 = 8  *0.01

NA = 100
coupon = 8  *0.01*NA

coupon/(1+z1)+ coupon/(1+z2)**2 + coupon/(1+z3)**3 + NA/(1+z3)**3

# 22
(6+ 6/(1+0.07)+ 106/(1+0.07)**2)/100-1


#


#%%
# 

z1 = 5
z2 = 6
z3 = 7
z4 = 8
z5 = 9
C = 10
FV = 1

DF_5 = 1/(1+z1*0.01) + 1/(1+z2*0.01)**2+ 1/(1+z3*0.01)**3+ 1/(1+z4*0.01)**4+ 1/(1+z5*0.01)**5

PV = C*0.01*DF_5 + FV*1/(1+z5*0.01)**5


#%%
# Derivative currency swap P54s
r_a = 2.7695*0.01
r_b = 0.2497*0.01

DF_sum_a = 3.967683
DF_sum_b = 3.994841
NA_a = 100000000
NA_b = 87719298
PV_n_a = 0.986031
PV_n_b = 0.998336
AP = 90/360
S_t = 1.13
V_cs = NA_a*(r_a*AP*DF_sum_a + PV_n_a) - NA_b*(r_b*AP*DF_sum_b + PV_n_b)*S_t

V_cs_b = -V_cs/S_t

#%%
# Derivative equity swap P61
# receive fix pay equity
AP = 180/360
z = 1.2
z1 = z*(1-AP)
z2 = z*(2-AP)
z3 = z*(3-AP)
z4 = z*(4-AP)
z5 = z*(5-AP)
C = 10000000*0.015
Par = 10000000
NA = Par
S_t = 105
S_t_prev = 100


DF_5 = 1/(1+z1*0.01) + 1/(1+z2*0.01)+ 1/(1+z3*0.01)+ 1/(1+z4*0.01)+ 1/(1+z5*0.01)
V_fix = C*DF_5+Par/(1+z5*0.01)

#DF_5 = 1/(1+z*0.01)**(1-AP) + 1/(1+z*0.01)**(2-AP)+ 1/(1+z*0.01)**(3-AP)+ 1/(1+z*0.01)**(4-AP)+ 1/(1+z*0.01)**(5-AP)
#V_fix = C*DF_5+Par/(1+z5*0.01)

V_eq_t = V_fix - S_t/S_t_prev*NA

S_t_prime = V_fix*S_t_prev/NA

#%%
# Derivative equity swap P66
# 
import numpy as np
AP = 90/360

z1 = 1.9
z2 = 2
z3 = 2.1
z4 = 2.2

array_spot_rate = [1.9,2.00,2.10,2.20]
array_days_maturity = [90,180,270,360]

def r_fix(array_spot_rate, array_days_maturity, AP, daycount = '30/360'):
    string_tuple = daycount.split('/')
    
    dc_month, dc_year = tuple(int(s) for s in string_tuple)
    arr_time = np.array(array_days_maturity)
    arr_time_adj = arr_time/dc_year
    arr_spot = np.array(array_spot_rate)*0.01
    print(dc_month)
    print(dc_year)
    print(arr_time/dc_year)
    print(arr_spot)
    #r*AP* sum(1/(1 + arr_spot*arr_time_adj))+ 1/(1+(arr_spot*arr_time_adj)[-1])
    
    
    return (arr_spot*arr_time_adj)[-1]/(1+(arr_spot*arr_time_adj)[-1])/sum(1/(1 + arr_spot*arr_time_adj))/AP

array_spot_rate = [1.9,2.00,2.10,2.20]
array_days_maturity = [90,180,270,360]
r_fix(array_spot_rate, array_days_maturity,AP= 90/360, daycount = '30/360')
    
    
array_spot_rate = [3,6,8]
array_days_maturity = [1,2,3]
r_fix(array_spot_rate, array_days_maturity,AP= 1, daycount = '30/1') 
    
    
    
    
2*8*100/(1+0.06*0.01)
    
# Q7 p67
# Q*CF  vs FV(B_0 + AI_0) - FVCI
(112+0.08)*(1+0.3*0.01)**(3/12)-0.2
# Q*CF
0.9*125 
# profit 
  (112+0.08)*(1+0.3*0.01)**(3/12)-0.2 - 0.9*125 

# Q8 p68

16080* exp((0.2996*0.01-1.1*0.01)*3/12)
 ,

# Q9 p68
250* exp((0.2996*0.01)*9/12)



# Q10 p69
245* exp((0.325*0.01)*6/12)- 1.5* exp((0.325*0.01)*3/12)
250.56238159910416 - 243.89722940659885


PV_list = np.array([0.990099,0.977876,0.965136])
(1-0.965136)/ sum(PV_list)

50000000*(1.12-3)*0.01*(0.990099+0.977876)


# Q14

PV_SUM = (1/(1+0.05*0.01*90/360) + 1/(1+0.1*0.01*180/360) +1/(1+0.15*0.01*270/360) +1/(1+0.25*0.01*360/360))
AP = 90/360
PV_n = 1/(1+0.25*0.01*360/360)
r =( 1-PV_n)/PV_SUM/AP


# Q15
PV_SUM =0.998004+0.985222+0.970874+0.934579+0.895255
AP = 1
PV_n = 0.895255

((2*0.01*AP*PV_SUM+PV_n)  -103/100)*20000000
((2*0.01*AP*PV_SUM+PV_n) )*100

# Q17

((1+6/12*0.95*0.01)/(1+3/12*0.9*0.01)-1)*12/3


r =( 1-PV_n)/PV_SUM/AP




# Q18
((1+4/12*0.92*0.01)/(1+0.75*0.01*1/12) -1)*12/3

# Q19
((1+ 0.94*0.01*5/12)/(1+0.82*0.01*2/12) -1)*12/3

# Q20
(1.1-0.7)*0.01*20000000*3/12/(1+1.1*0.01*3/12)
















