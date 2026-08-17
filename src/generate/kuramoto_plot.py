from platform import node

from matplotlib.pylab import False_, eig
from numpy._core.arrayprint import format_float_scientific

from plot_utils import *
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
plt.rcParams['text.usetex'] = True
"""
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'  # Use LaTeX's default serif font
plt.rcParams['text.latex.preamble'] = r'\\usepackage{amsmath}'  # Optional: Add LaTeX packages
"""

import numpy as np
from kuramoto_class import SecondOrderKuramotoModel as sokm 
from shift_matrix import *
import networkx as nx
import os
import svgutils.transform as sg
import svgutils.compose as sc
from cairosvg import svg2png

def generate_frequency_range(model:sokm,extra_scope=0.2):
    """
        Helper function to generate a frequency range for the domain over which the resonant behavior is plotted.

        Parameters
        ----------
        model : sokm
            The network model containing the second order Kuramoto model Jacobian and system dynamics.
        extra_scope : floar, default=0.2
            Extra frequency range padding (relative to resonant region) when auto-generating the frequency range.
        
        Returns
        -------
        np.ndarray
            Array of frequency values over which to plot the resonant behavior, automatically generated based on the model's predicted resonance frequencies with extra_scope padding.
    """

    resonance_frequencies=model.predict_resonance_frequencies()
    resonance_frequencies=resonance_frequencies[np.invert(np.isnan(resonance_frequencies))] #throwing out the np.nan values caused by 0 eigenvalue
    lower_bound=np.min(resonance_frequencies)
    upper_bound=np.max(resonance_frequencies)

    puffer=extra_scope*(upper_bound-lower_bound)
    lower_bound-=puffer
    upper_bound+=puffer
    return np.linspace(lower_bound,upper_bound,num=1000) #frequency values over which we plot

   
def plot_network(
    model: sokm,
    shift_matrix_obj: Optional[ShiftMatrix]= None,
    seed=42,
    edge_weight_key="weight",
    edge_weight_visualization="length",
    vtn_edge_style="dotted",
    vtn_color=None,
    name="network", 
    fs=6,
    threshhold=1e-10,
    perturbed_node=None,
    node_colors=None,
    axes=None,
    node_size=10,
    labels=True,
    show_node_type=False,
):
    """
    Plot a network with edge weights represented as thickness or as geometry (edge length), plus an
    optional VTN visualization layered on top.

    Parameters
    ----------
    model : sokm
        The network model containing the connectivity matrix.
    shift_matrix_obj : Optional[ShiftMatrix], default=None
        Optional shift matrix object for identifying nodes requiring active control. If provided,
        these nodes are highlighted with colors corresponding to individual shifts.
    seed : int, default=42
        Seed for reproducibility of the spring layout. Default is 42.
    edge_weight_key : str, default="weight"
        The key in the edge attributes that represents the weight. Default is "weight".
    edge_weight_visualization : str, default="width"
        How to visualize edge weights. "width" scales edge thickness by weight; "length" uses the
        edge weights in the spring layout so stronger connections appear closer together.
    vtn_edge_style : str, default="dashed"
        Style for the edges connecting the highlighted nodes. Default is "dashed".
    name : str, default="network.svg"
        Filename for the saved plot.
    fs : int, default=6
        Fontsize for node labels.
    threshhold : float, default=1e-10
        Threshold for identifying non-zero entries in the shift matrix rows that indicate
        nodes requiring active control.
    perturbed_node : int, optional
        Index of the perturbed node. Currently reserved for future use. Default is None.

    Returns
    -------
    str
        Path to the saved plot file.
    """
    if edge_weight_visualization not in ("width", "length","color","log_color"):
        raise ValueError("edge_weight_visualization must be 'width' or 'length' or 'color' or 'log_color'")

    # Create the graph from the connectivity matrix
    G = nx.from_numpy_array(model.connectivity_matrix)

    # Extract edge weights
    edge_weights = [d.get(edge_weight_key, 1.0) for _, _, d in G.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1.0
    edge_colors="black"

    layout_weight = None
    edge_widths = [1.0 for _ in edge_weights]

    # Normalize edge widths for visual representation
    if edge_weight_visualization == "length":
        layout_weight = edge_weight_key
    elif edge_weight_visualization == "width":
        edge_widths = [2 * (w / max_weight) for w in edge_weights]  # Scale edge thickness
    elif edge_weight_visualization == "color":
        cmap= create_cmap_from_white(black, cmap_length=256)
        edge_colors = [cmap(w / max_weight) for w in edge_weights]  # Scale color
    elif edge_weight_visualization == "log_color": # DOES NOT WORK YET
        cmap= create_cmap_from_white(black, cmap_length=256)
        #print("normalized edge weights:", [w / max_weight for w in edge_weights] )
        edge_colors = [cmap(np.log(w / max_weight)) for w in edge_weights]  # Scale color
        #edge_opacities = [0.2 + 0.8 * (w / max_weight) for w in edge_weights]  # Scale opacity
    else:
        edge_colors = "black"

    pos = nx.spring_layout(G, seed=seed, weight=layout_weight)
    # Create the figure and axis
    if axes== None:
        if show_node_type:
            fig, ax = plt.subplots(figsize=(4., 1.6))
        else:
            fig, ax = plt.subplots(figsize=(2., 0.8))
    else:
        ax=axes

    #node_color=darkblue+(1.-darkblue)*0.35
    if node_colors is None:
        node_color=node_colors_from_perturbation_distance(perturbed_node, model, cmap=plt.cm.cividis) #coloring nodes according to their distance to the perturbed node. The closer the node the more intense the color. Nodes further than max_distance are colored with the base color
    else:
        node_color=node_colors
    

    # Draw the base graph
    if show_node_type:
        p=model.power_vector
        square=np.where(p>0)[0]
        triangle=np.where(p<0)[0]
        nx.draw(
            G,
            pos,
            nodelist=[],
            ax=ax,
            with_labels=labels,
            node_size=node_size,
            node_shape="s",
            node_color="none",
            font_size=fs,
            font_color="black",
            edge_color=edge_colors,
            width=edge_widths,
            linewidths=0.5,
            alpha=1.,  # Base opacity for edges
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=triangle,
            ax=ax,
            node_size=node_size,
            node_shape="^",
            node_color="w",
            edgecolors=edge_colors,
            linewidths=0.5,
            alpha=1.,  # Base opacity for edges
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=square,
            ax=ax,
            node_size=node_size,
            node_shape="s",
            node_color="w",
            edgecolors=edge_colors,
            linewidths=0.5,
            alpha=1.,  # Base opacity for edges
        )
        ax.annotate(
        r"$\textbf{a1}$",
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.1, -0.1), textcoords='offset fontsize',
        fontsize=fs, verticalalignment='top', fontfamily='serif')
    else:
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=labels,
            node_size=node_size,
            node_color=node_color,
            font_size=fs,
            font_color="black",
            edge_color=edge_colors,
            width=edge_widths,
            linewidths=2,
            alpha=1.,  # Base opacity for edges
        )

    # Drawing vtn edges if a shift matrix obj or shift matrix are provided
    plot_vtn = type(shift_matrix_obj) is ShiftMatrix or type(shift_matrix_obj) is np.ndarray
    if plot_vtn:
        
        if vtn_color is None:
            # fetch individual shift matrices and their color coding
            individual_shift_matrices=shift_matrix_obj.cunstruct_individual_shift_matrices(in_eigenspace=False)
            n_individual_shift_matrices=len(individual_shift_matrices)
            shift_colors=create_shift_colors(n_individual_shift_matrices) 
        else:
            n_individual_shift_matrices=1
            individual_shift_matrices=np.array([shift_matrix_obj])
        

        for shift_number in range(n_individual_shift_matrices):

            # collecting all non zero row indices of the individual shift aka the nodes which need active control
            vtn_nodes= np.nonzero(np.any(individual_shift_matrices[shift_number] > threshhold, axis=1))[0]

            # Fully connect the vtn nodes with dashed edges
            for i, node1 in enumerate(vtn_nodes):
                for node2 in vtn_nodes[i + 1 :]:
                    ax.plot(
                        [pos[node1][0], pos[node2][0]],
                        [pos[node1][1], pos[node2][1]],
                        linestyle=vtn_edge_style,
                        color=shift_colors[shift_number] if vtn_color is None else vtn_color,
                        alpha=1.,
                        linewidth=np.average(edge_widths)*2
                    )

        # Highlight vtn nodes if provided
        
        # fetch individual shift matrices and their color coding
        if False:
            individual_shift_matrices=shift_matrix_obj.cunstruct_individual_shift_matrices(in_eigenspace=False)
            n_individual_shift_matrices=len(individual_shift_matrices)
            shift_colors=create_shift_colors(n_individual_shift_matrices) 

        if vtn_color is None:
            for i in shift_colors:
                shift_colors[i][3]=1.

        for shift_number in range(n_individual_shift_matrices):

            # collecting all non zero row indices of the individual shift aka the nodes which need active control
            vtn_nodes= np.nonzero(np.any(individual_shift_matrices[shift_number] > threshhold, axis=1))[0]

            # overlying the shift color for the vtn nodes
            nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=vtn_nodes,
                    edgecolors=list(shift_colors[shift_number]) if vtn_color is None else vtn_color,
                    node_color=[0.,0.,0.,0.],
                    node_size=node_size+15,
                    ax=ax,
                )

    """
    # highlighting the perturbing node
    if perturbed_node is not None:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[perturbed_node],
                edgecolors="grey",
                node_size=100,
                ax=ax,
                node_color="none",
            )
    """

    

    if axes== None:
        # Save the figure 
        if model.current_dir == None:
            model.save_parameters()
        
        if shift_matrix_obj!=None:
            save_dir=shift_matrix_obj.current_dir
        else:
            save_dir = model.current_dir
        svg_path=save_figure(fig, save_dir=save_dir, name=name+".svg")
        png_path=os.path.join(save_dir,name+".png")
        svg2png(url=svg_path,write_to=png_path,
                parent_height=140,parent_width=130,output_height=140*4,output_width=130*4)


        return svg_path


def break_axes(axes, size=6):
    axes[0].spines.right.set_visible(False)
    axes[1].spines.left.set_visible(False)
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=size,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axes[0].plot([1, 1], [1, 0], transform=axes[0].transAxes, **kwargs)
    axes[1].plot([0, 0], [0, 1], transform=axes[1].transAxes, **kwargs)


def network_w_response(scenario, 
                       model:sokm, 
                       t_final, 
                       steps, 
                       min_max=None,
                       y_0=None, 
                       scenario_name="test", 
                       overwrite=True, 
                       vtn_on=True, 
                       only_perturbation=False, 
                       node=0, 
                       t_window=200,
                       save_dir=None, 
                       linewidth=1.,
                       response_max=None,
                       plot_real_psd=True,
                       cmap= "coolwarm",
                       axes=None,
                       color=None,
                       labels=False,
                       panel="a",
                       fontsize=10,
                    euler_maruyama=False,
                    onset_steady=500):
    shift_matrix_obj = scenario[3][0] if vtn_on else None
    pert_amplitude, pert_freq, pert_node, onset_t= scenario[1]

    transient = len(axes)==3
    t_window_steady= t_window/2 if transient else t_window

    # create grid of subplots with first axis narrower than the others
    save=False
    if axes is None:
        save = True
        fig, axes = plt.subplots(1, 2, figsize=(2.8, 0.6), gridspec_kw={"width_ratios": [3, 3]})
    
    if type(color) == tuple and len(color) > 1:
        scenario_color=blend_colors(color,mode="plt")
    else:
        scenario_color=color

    # calculate trajectories
    traj=generate_or_fetch_scenario_data(t_final, scenario, model, y_0=y_0, steps=steps, meta_scenario_name=scenario_name, save_dir=save_dir, overwrite=overwrite, minus_fixpoint= True,silent=True,
            euler_maruyama=euler_maruyama)
    t=traj["t"]
    if vtn_on:
        vals=traj["ys_shifted"][:np.shape(model.jacobian_matrix)[0],:] #fetching only the node trajectories, not the velocity trajectories
    else:
        vals=traj["ys"][:np.shape(model.jacobian_matrix)[0],:] #fetching only the node trajectories, not the velocity trajectories
    if only_perturbation:
        vals=np.zeros((np.shape(model.jacobian_matrix)[0],steps+1))
        vals=dynamics.sine_perturbation_single_node(t,vals[:,0],scenario[1])
        node = pert_node # setting node i.e. the of vals which we plot to the index of perturbed node
    # calculate steady state response psd for each node
    t_traj_index=np.abs(t - onset_steady).argmin() 
    freqs, psd = compute_psd_from_traj(t[t_traj_index:], vals[:,t_traj_index:]) 
    freqs =2*np.pi*freqs # converting to omega



    # plotting network
    if not only_perturbation:
        ax_network = axes[0].inset_axes([0.2, 0.2, 0.6, 0.75])

        if False:
            # multiply with perturbation psd to get color coding of nodes according to their response to the driving signal, or just np.max
            idx = np.argmin(np.abs(freqs - pert_freq)) #finding the frequency closes to the driving frequency
            response_amplitude= psd[:,idx] # for visual quantification of response strength taking the values corresponding to that frequency

        response_amplitude = np.max(np.abs(vals[:,t_traj_index:]), axis=1) 
        #response_amplitude = np.max(psd, axis=1)
        if response_max is None:
            response_max= np.max(response_amplitude)

        if cmap=="RdYBlu_r":
            response_amplitude = response_amplitude / (2*response_max)+0.5 # Normalize to [0.5,1.]
            node_colors = plt.cm.RdYlBu_r(response_amplitude)
        elif cmap== "coolwarm":
            response_amplitude = response_amplitude / (response_max) # Normalize to [0.5,1.]
            node_colors = plt.cm.coolwarm(response_amplitude)

        if plot_real_psd:
            #for i in range(np.shape(psd)[0]):
            #    axes[1+transient].plot(freqs,psd[i,:],linewidth=linewidth,color=node_colors[i])
            axes[1+transient].plot(freqs,psd[node,:],linewidth=linewidth,color=node_colors[node])

        # plot network with node color coding according to response psd
        plot_network(
            model,
            shift_matrix_obj= shift_matrix_obj,
            seed=42,
            vtn_edge_style=(0,(1,1)),
            vtn_color=scenario_color,
            #vtn_color="#294e62ff",
            name="network.svg", 
            fs=10,
            threshhold=1e-10,
            perturbed_node=None,
            node_colors=node_colors,
            #axes=axes[1],
            axes=ax_network,
            node_size=10,
            labels=labels
        )

    # plot trajectory for one selected node
    t_traj_index=np.abs(t - t_final + t_window_steady).argmin() # find t closest to t_final bases on steps
    t_window_index=np.abs(t - t_window).argmin() # find t closest to t_window bases on steps
    if color is None:
        color=orange
    
    if transient:
        axes[0].plot(t[:t_window_index], vals[node,:t_window_index], color=scenario_color if only_perturbation else node_colors[node],linewidth=linewidth)
        axes[1].set_yticks([])
        axes[0].set_xlim(np.min(t),t_window)
    axes[0+transient].plot(t[t_traj_index:], vals[node,t_traj_index:], color=scenario_color if only_perturbation else node_colors[node],linewidth=linewidth)
    axes[0+transient].set_xlim( t_final - t_window_steady,t_final)
    if min_max==None:
        if only_perturbation:
            min_max=(2.*min(vals[node,t_traj_index:]),2.*max(vals[node,t_traj_index:]))
        else:
            min_max=(1.2*min(vals[node,t_traj_index:]),1.2*max(vals[node,t_traj_index:]))
    axes[0].set_ylim(min_max)
    if transient:
        axes[1].set_ylim(min_max)
        axes[0].set_xticks([ np.round(np.min(t)) , np.round(onset_t), np.round(0.8*t_window) ])
    axes[0].set_yticks(np.round(min_max if only_perturbation else (0.,min_max[1]),decimals=1))

    axes[0+transient].set_xticks([np.round(t_window_steady*0.2+np.min(t[t_traj_index:])),np.max(t[t_traj_index:])])

    

    # calculate theoretical psd of that node and plot it
    omega=generate_frequency_range(model,extra_scope=0.2)
    

    if only_perturbation:
        if np.isscalar(pert_freq):
            axes[1+transient].vlines(pert_freq,0,pert_amplitude, color="black", linewidth=linewidth,alpha=1)
        else:
            for i, freq in enumerate(pert_freq):
                axes[1+transient].vlines(freq,0,pert_amplitude if type(pert_amplitude) is float else pert_amplitude[i], color="black", linewidth=linewidth,alpha=1)
        
    else:
        S = None
        if vtn_on:
            if type(shift_matrix_obj) is ShiftMatrix:
                S=shift_matrix_obj.shift_matrix
            else:
                S=shift_matrix_obj
        response_vals=model.calculate_response_amplitudes(omega,S=S,k=pert_node)
        axes[1+transient].plot(omega, response_vals[node], color="black",linewidth=linewidth)
        axes[1+transient].set_ylim((np.min(response_vals[node]) if np.min(response_vals[node]) > 1e-4 else 1e-4, 10*np.max(response_vals[node])))
    axes[1+transient].set_xlim(min(omega),max(omega))    
    axes[1+transient].set_yscale("log")
    #axes[1+transient].yaxis.tick_right()
    if not only_perturbation:
        axes[1+transient].set_ylim(0.5*np.min(response_vals),np.max(response_vals)*2) #<- this is weird, probably redundant
        #axes[1+transient].set_yticks([round_log(np.min(response_vals[node])),round_log(np.max(response_vals[node])*2)])
        #axes[1+transient].set_yticks([1e-2,1e2])
    else:
        axes[1+transient].set_ylim(1e-10,np.max(pert_amplitude)*2)
        axes[1+transient].set_yticks([1e-10,np.max(pert_amplitude)*10])
    if plot_real_psd and not only_perturbation:
        axes[1+transient].set_ylim((1e-10,10*np.max(response_vals[node])))
    
    # addid perturbation prequency patches
    delta=0.05
    if np.isscalar(pert_freq):
        rect=plt.Rectangle((pert_freq-delta,0), 2*delta, 1e3, color=color,alpha=0.3,edgecolor=[0,0,0,0],linewidth=linewidth)
        axes[1+transient].add_patch(rect)
    else:
        for i, freq in enumerate(pert_freq):
            if len(pert_freq)<=len(color):
                rect=plt.Rectangle((freq-delta,0), 2*delta, 1e3, color=color[i],alpha=0.3,edgecolor=[0,0,0,0],linewidth=linewidth)
                axes[1+transient].add_patch(rect)

    if only_perturbation:
        if np.isscalar(pert_freq):
            axes[1].vlines(pert_freq,0,pert_amplitude, color="black", linewidth=linewidth,alpha=1)
        else:
            for i, freq in enumerate(pert_freq):
                axes[1].vlines(freq,0,pert_amplitude if type(pert_amplitude) is float else pert_amplitude[i], color="black", linewidth=linewidth,alpha=1)

    label_offset=0
    if only_perturbation is False and vtn_on is False:
        label_offset=2
    elif only_perturbation is False and vtn_on:
        label_offset=4
    
    label_1= panel + f"{str(1+label_offset)}"
    axes[0].annotate(rf"\textbf{{{label_1}}}",
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.1, -0.1), textcoords='offset fontsize',
        fontsize=fontsize, verticalalignment='top', fontfamily='serif')
    label_2= panel + f"{str(2+label_offset)}"
    axes[1+transient].annotate(rf"\textbf{{{label_2}}}",
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.1, -0.1), textcoords='offset fontsize',
        fontsize=fontsize, verticalalignment='top', fontfamily='serif')
    
    # breaking axes
    if transient:
        break_axes(axes)

    # saving as SVG and PNG
    if save:
        if save_dir is None:
            if shift_matrix_obj is not None:
                save_dir = shift_matrix_obj.current_dir
            else:
                shift_matrix_obj = scenario[3][0]
                save_dir = shift_matrix_obj.current_dir
        if only_perturbation:
            specific_name=scenario_name+"_perturbation"
        else:
            specific_name=scenario_name+"_vtn" if vtn_on else scenario_name+"_no_vtn"
        if euler_maruyama:
            specific_name+="_euler_maruyama"
        svg_path=save_figure(fig, save_dir=save_dir, name=specific_name+".svg")
        png_path=os.path.join(save_dir,specific_name+".png")
        svg2png(url=svg_path,write_to=png_path,
                    parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)
        if not only_perturbation and not vtn_on:
            return (response_max, svg_path)
        else:
            return svg_path
    else:
        return response_max


def vtn_scenario_powers(t_final,t_window,scenario,model,axes,absolute=False, save_dir=None, threshhold=1e-10,epsilon=None,specific_name="shift",fontsize=10, linewidth=1., panel="a" ):
    """
    Plots the vtn of the powers of a given scenario for the on and off case in a format compatible with plot network_w_response. 
    color: 294e62ff
    """

    transient = len(axes)==3
    t_window_steady= t_window/2 if transient else t_window

    shift_matrix_obj=scenario[3][0]

    #fig, axes = plt.subplots(3, 1, figsize=(1.4, 2.6)) 
    #axes[1].set_axis_off()
    #plt.subplots_adjust(hspace=0.25, wspace=0.05)

    # fetching indices nodes where powers are applied based on the shift matrix
    if type(shift_matrix_obj) is ShiftMatrix:
        S=shift_matrix_obj.shift_matrix
    else:
        S=shift_matrix_obj
    vtn_nodes= np.nonzero(np.any(np.abs(S) > threshhold, axis=1))[0]

    # fetching trajectory data and fixpoint values
    traj=generate_or_fetch_scenario_data(t_final, scenario, model, minus_fixpoint= True, save_dir=save_dir )
        
    t=traj["t"]
    y_fixpoint=model.compute_fixed_point()
    ys_shifted=traj["ys_shifted"][:len(y_fixpoint),:]

    # compute shift power from that
    p_vtn= (S@ys_shifted)[vtn_nodes,:]
    p_vtn_average= np.cumsum(p_vtn,axis=1)[:,1:]/t[None,1:] #skipping first entry in order to avoid division by zero

    # vtn_off
    #p_vtn_off= np.zeros_like(p_vtn_average)

    if absolute:
        p_vtn_average=np.abs(p_vtn_average)

    p_i=np.max(model.power_vector)
    p_vtn_average=p_vtn_average/p_i
    # bess colors
    #colors=["orange", "blue","green","red","darkblue"]
    colors=[]

    # plot
    t_traj_index=np.abs(t - t_final + t_window_steady).argmin()
    t_window_index=np.abs(t - t_window).argmin()
    
    for bess_index in range(len(vtn_nodes)):
        #axes.plot(t,p_vtn[bess_index], color=colors[bess_index],linewidth=0.5)
        if transient:
            axes[0].plot(t[:t_window_index],p_vtn[bess_index,:t_window_index],color=colors[bess_index] if len(colors) > bess_index else "#294e62ff",linewidth=linewidth, alpha=0.8)
            axes[0].plot(t[:t_window_index],np.zeros_like(t[:t_window_index]),color="black",linewidth=linewidth/2, alpha=0.8, linestyle ="--")
        axes[0+transient].plot(t[t_traj_index:],p_vtn[bess_index,t_traj_index:],color=colors[bess_index] if len(colors) > bess_index else "#294e62ff",linewidth=linewidth, alpha=0.8)
        axes[0+transient].plot(t[t_traj_index:],np.zeros_like(t[t_traj_index:]),color="black",linewidth=linewidth/2, alpha=0.8, linestyle ="--")
        axes[1+transient].plot(t[1:],p_vtn_average[bess_index],color=colors[bess_index] if len(colors) > bess_index else "#294e62ff", linewidth=linewidth, alpha=0.8)
        axes[1+transient].plot(t[1:],np.zeros_like(t[1:]),color="black",linewidth=linewidth/2, alpha=0.8, linestyle ="--")
        #plt.plot(t,p_vtn[bess_index], color=colors[bess_index])
        #plt.plot(t[1:],p_vtn_average[bess_index], linestyle="--",color=colors[bess_index])
        #plt.show()

    if absolute:
        axes.set_yscale("log")
    #axes[1].set_xticks([])
    if transient:
        axes[0].set_xlim(np.min(t),t_window)
        axes[0].set_xticks([ np.round(np.min(t)) , np.round(scenario[1][3]) , np.round(0.8*t_window) ])
    axes[0+transient].set_xlim((np.round(t[t_traj_index]),t_final))
    axes[0+transient].set_xticks([np.round(t_window_steady*0.2+np.min(t[t_traj_index:])),np.max(t[t_traj_index:])])
    axes[1+transient].set_xlabel(r"$t$")
    axes[1+transient].set_xlim(np.round((min(t),t_final)))

    # y_axis labeling
    y_lims=(np.min(p_vtn),np.max(p_vtn))
    if np.min(p_vtn)>-0.1 and np.max(p_vtn)<0.1: # avoiding too long labels for the plot
        y_lims= (-0.1,0.1)
    y_ticks=round_log(np.array(y_lims),decimals=0)

    if transient:
        axes[0].set_ylim(y_lims)
        axes[1].set_yticks([])
    
    axes[0+transient].set_ylim(y_lims)
    axes[1+transient].set_ylim(y_lims)
    #y_vals_avrg=np.append(y_lims_avrg,0.)
    #y_ticks=[rf"${{{val}}}\%$" for val in np.round(y_vals_avrg*100,decimals=2)]
    #axes[1].set_yticks(y_vals,labels=y_ticks)
    #axes[2].set_yticks(y_vals_avrg,labels=y_ticks)
    axes[0].set_ylabel(r"$P_{\mathrm{s}}/{P_{\mathrm{p}}}$")
    axes[0].set_yticks(y_ticks)
    axes[1+transient].set_yticks(y_ticks)
    axes[1+transient].set_ylabel(r"$\overline{P}_{\mathrm{s}}/{P_{\mathrm{p}}}$")
    #axes[2].yaxis.set_label_position("right")
    #axes[2].yaxis.tick_right()

    # create the broken axes effect
    if transient: 
        break_axes(axes)

    label_1= panel + f"{str(7)}"
    label_2= panel + f"{str(8)}"
    axes[0].annotate(
        rf"\textbf{{{label_1}}}",
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.1, -0.1), textcoords='offset fontsize',
        fontsize=fontsize, verticalalignment='top', fontfamily='serif')
    axes[1+transient].annotate(
        rf"\textbf{{{label_2}}}",
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.1, -0.1), textcoords='offset fontsize',
        fontsize=fontsize, verticalalignment='top', fontfamily='serif')
    #axes[1].annotate(r"$(\mathrm{a})$", (-0.5, 0.5), fontsize=fontsize, annotation_clip=False)

    # saving
    if False:
        if save_dir is None:
            save_dir = shift_matrix_obj.current_dir
        specific_name+= "cumsum_vtn_powers"
        svg_path=save_figure(fig, save_dir=save_dir, name=specific_name+".svg")
        png_path=os.path.join(save_dir,specific_name+".png")
        svg2png(url=svg_path,write_to=png_path,
                    parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)
    #return svg_path


def individual_shift_scenarios(shift_matrix_obj, pert_amplitude, pert_node, flip=False, onset_t=0.):
    """
    Helper function that generates a sine perturbation scenario for each individual shift matrix in shift_matrix_object.
    """

    # fetch individual shift matrices
    permutation, individual_shift_matrices=shift_matrix_obj.construct_individual_shift_matrices(space="physical")

    # collect original resonance frequencies prio to shift for perturbation
    all_orig_freqs = model.predict_resonance_frequencies()
    all_orig_freqs = all_orig_freqs[np.invert(np.isnan(all_orig_freqs))]

    # eigenvalue indices of the shifted eigenvalues
    eigval_indices = shift_matrix_obj.eigenvalue_indices

    # flipping order for plotting if desired
    if flip:
        individual_shift_matrices=np.flip(individual_shift_matrices, axis=0)
        all_orig_freqs=np.flip(all_orig_freqs)
        eigval_indices=len(all_orig_freqs)-1-np.flip(eigval_indices) # the -1 is necessary because array indexing starts at 0

    individual_scenarios=[]

    # loop over individual shift matrices and create a scenario for ich with a sine perturbation corresponding to the eigenvalue shifted by the individual shift matrix
    for i, ism in enumerate(individual_shift_matrices):
        shift_args = (ism, model, None)
        shift = (dynamics.jacobian_shift, shift_args)
        pert = (dynamics.sine_perturbation_single_node, (pert_amplitude,all_orig_freqs[eigval_indices[i]],pert_node, onset_t))
        individual_scenarios.append(pert+shift)

    if len(individual_shift_matrices)>1:
        shift_args = (shift_matrix_obj.shift_matrix, model, None)
        shift = (dynamics.jacobian_shift, shift_args)
        pert_freqs=all_orig_freqs[eigval_indices]
        pert = (dynamics.sine_perturbation_single_node, (pert_amplitude,pert_freqs,pert_node,onset_t))
        individual_scenarios.append(pert+shift)

    return individual_scenarios


def scenario_panel_recursive(model,
                             scenario,
                             t_final,
                             steps=16000,
                             color=np.array(orange), 
                             save_dir=None, 
                             specific_name="scenario_panel", 
                             overwrite=True,
                             node=0, 
                             pert_node=1,
                             labels=False,
                             fontsize=10,
                             plot_real_psd=False,
                             amplitude=0.1,
                             y_0=None,
                             t_window=50,
                             panel="a",
                             cmap="coolwarm",
                             linewidth=1.,
                             transient=True, 
                             onset_t = 1.,
                             euler_maruyama=False
                             ):
    
    # allows for calling with a specific scenario, but also just with a shift matrix object
    print("scenario:", scenario)
    if isinstance(scenario, ShiftMatrix): #if no scenario, but only the shiftMatrix object (class) is provided the object construction is visualized  the individual scnearios generated
        shift_matrix_obj=scenario
        compose_shift_matrix_construction_visualization_horizontal(shift_matrix_obj, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=overwrite)
        scenario_provided=False
        check_individual_shifts=True
    else:
        shift_matrix_obj=scenario[3][0]
        if isinstance(shift_matrix_obj, ShiftMatrix):
            compose_shift_matrix_construction_visualization_horizontal(shift_matrix_obj, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=overwrite)
            check_individual_shifts=True
        else: # case of scenario with np.array shift matrix instead of ShiftMatrix object (class), sorry for poor naming of stuff
            check_individual_shifts = False
        scenario_provided=True
    
    # check if individual scenarios need to be generated#
    if check_individual_shifts:
        if len(shift_matrix_obj.shifts)>1 or scenario_provided is False:

            # generate individual shift scenarios, the corresponding colors
            individual_scenarios= individual_shift_scenarios(shift_matrix_obj,amplitude,pert_node=pert_node, onset_t=onset_t)
            #individual_scenarios=individual_scenarios[::-1] # reverse order for plotting
            n_scenarios=len(individual_scenarios)
            scenario_colors=create_shift_colors(n_scenarios-1)

            for i in range(n_scenarios-1):
                scenario_colors[i][3]=1.
            
            # common shift directory for all individual scenarios
            if save_dir is None:
                save_dir= shift_matrix_obj.current_dir

            # iterate through them recursively
            for i, individual_scenario in enumerate(individual_scenarios):
                print(f"Now working on individual shift scenario {i}.")
                scenario_panel_recursive(model,
                                        individual_scenario,
                                        t_final, 
                                        steps=steps, 
                                        save_dir=save_dir, 
                                        specific_name=f"individual_shift_{i}" if i < len(individual_scenarios)-1 else f"full_shift",
                                        overwrite=overwrite, 
                                        node=node,
                                        color=np.array(scenario_colors[i]) if i < len(scenario_colors) else np.array(scenario_colors),
                                        fontsize=fontsize,
                                        plot_real_psd=plot_real_psd,
                                        labels=labels,
                                        panel=chr(98+n_scenarios-i-2) if (i<n_scenarios-1 or n_scenarios==1) else chr(97+3*(n_scenarios+1)),
                                        t_window=t_window,
                                        linewidth=linewidth,
                                        transient=True if i == 1 else False)

    # plot scenario of the full shift matrix if provided
    if scenario_provided:
        #fig, axes = plt.subplots(4, 3, figsize=(3.6, 4.)) #, gridspec_kw={"width_ratios": [3, 3]}) ehem. (2.7,2.1)

        def create_scenario_panel_axes(transient=False):
            """
            Creates the panel axes and figure instance for the plot. If transient is True three axes has the shape (4 x 2) otherwise (3 x 2) and the first column is dropped.
            """


            # creating main grid for plots
            if transient:
                fig = plt.figure(figsize=(4.4, 3.2))
                gs_main = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.3,height_ratios=[6,2], width_ratios=[2,1])
            else:
                fig = plt.figure(figsize=(2.9, 3.2))
                gs_main = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.3*1.5,height_ratios=[6,2], width_ratios=[1,1])
            # creating subgrids
            hspace=0.15
            wspace=0.05
            if transient:
                gs_traj = gs_main[0,0].subgridspec(3, 2,hspace=hspace, wspace=wspace,width_ratios=[2,1])
                gs_p = gs_main[1,0].subgridspec(1, 2, wspace=wspace,width_ratios=[2,1])
            else:
                gs_traj = gs_main[0,0].subgridspec(3, 1,hspace=hspace, wspace=wspace)
                gs_p = gs_main[1,0].subgridspec(1, 1, wspace=wspace)
            gs_psd = gs_main[0,1].subgridspec(3, 1,hspace=hspace)
            gs_p_mean = gs_main[1,1].subgridspec(1, 1)

            # creating axes for plotting inside the subgrids
            axes = [[None for _ in range(2+transient)] for _ in range(4)]
            for i in range(3): 
                if transient:
                    axes[i][0]=fig.add_subplot(gs_traj[i, 0])
                axes[i][0+transient]=fig.add_subplot(gs_traj[i, 0+transient])
                axes[i][1+transient]=fig.add_subplot(gs_psd[i, 0])
            if transient:
                axes[3][0]=fig.add_subplot(gs_p[0, 0])
            axes[3][0+transient]=fig.add_subplot(gs_p[0, 0+transient])
            axes[3][1+transient]=fig.add_subplot(gs_p_mean[0, 0])
            return fig, axes
        
        fig, axes= create_scenario_panel_axes(transient)

        

        traj=generate_or_fetch_scenario_data(t_final, scenario, model, y_0=y_0, steps=steps, meta_scenario_name=specific_name, save_dir=save_dir, overwrite=overwrite, minus_fixpoint= True, silent=False)
        vals=np.zeros((np.shape(model.jacobian_matrix)[0],steps+1))
        vals=dynamics.sine_perturbation_single_node(traj["t"],vals[:,0],scenario[1])

            
        # y lims for trajectory plot
        mini=min(np.min(traj["ys"][node,:]),np.min(traj["ys_shifted"][node,:]))
        maxi=max(np.max(traj["ys"][node,:]),np.max(traj["ys_shifted"][node,:]))
        maxi+=(maxi-mini)*4
        min_max=(1.2*mini,1.2*maxi)

        network_w_response(scenario, 
                        model, 
                        t_final, 
                        axes=axes[0],  
                        color=color,
                        only_perturbation=True, 
                        y_0=y_0, 
                        steps=steps, 
                        scenario_name=specific_name, 
                        overwrite=False, 
                        vtn_on=False, 
                        node=node, 
                        t_window=t_window,
                        plot_real_psd=plot_real_psd,
                        labels=labels,
                        save_dir=save_dir,
                        panel=panel,
                        cmap=cmap,
                        linewidth=linewidth)
        response_max=network_w_response(scenario, 
                                model, 
                                t_final, 
                                axes=axes[1],   
                                color=color,
                                y_0=y_0, 
                                steps=steps, 
                                scenario_name=specific_name, 
                                overwrite=False, 
                                vtn_on=False, 
                                node=node, 
                                t_window=t_window,
                                min_max=min_max,
                                plot_real_psd=plot_real_psd,
                                labels=labels,
                        save_dir=save_dir,
                        panel=panel,
                        cmap=cmap,
                        linewidth=linewidth)
        network_w_response(scenario, 
                            model, 
                            t_final, 
                            axes=axes[2], 
                            color=color,
                            y_0=y_0, 
                            steps=steps, 
                            scenario_name=specific_name, 
                            overwrite=False, 
                            vtn_on=True,
                            response_max=response_max, 
                            node=node, 
                            t_window=t_window,
                            min_max=min_max,
                            plot_real_psd=plot_real_psd,
                            labels=labels,
                        save_dir=save_dir,
                        panel=panel,
                        cmap=cmap,
                        linewidth=linewidth)
        
        vtn_scenario_powers(t_final,t_window,scenario,model,axes[3],absolute=False,epsilon=amplitude,save_dir=save_dir,fontsize=fontsize,linewidth=linewidth, panel=panel)

        if transient:
            axes[0][2].set_xticks([])
            axes[1][2].set_xticks([])
        axes[0][0].set_xticks([])
        axes[0][1].set_xticks([])
        axes[1][1].set_xticks([])
        axes[1][0].set_xticks([])
        #axes[2,1].set_xticks([])
        #plt.subplots_adjust(hspace=0.25, wspace=0.05)

        # labels
        print("Computing LaTex labels.")
        #axes[0,0].annotate(r"$(\mathrm{b})$", (-100, 0), fontsize=fontsize, annotation_clip=False)
        axes[0][0].set_ylabel(r"$g_{k}(t)$",fontsize=fontsize,labelpad=0.1)
        axes[1][0].set_ylabel(r"$\Delta\theta_l$",fontsize=fontsize)
        axes[2][0].set_ylabel(r"$\Delta\theta_l$",fontsize=fontsize)
        axes[0][1+transient].set_ylabel(r"$\hat{g}_{k}(\omega)$",fontsize=fontsize)
        #axes[0][2].yaxis.set_label_position("right")
        axes[1][1+transient].set_ylabel(r"$\mathrm{A}_l$",fontsize=fontsize)
        #axes[1][2].yaxis.set_label_position("right")
        axes[2][1+transient].set_ylabel(r"$\mathrm{A}_l$",fontsize=fontsize)
        #axes[2][2].yaxis.set_label_position("right")
        #axes[2,1].xaxis.label.set_position((0.5, 1.1))
        axes[2][0].set_xlabel(r"$t$",fontsize=fontsize)
        axes[3][0].set_xlabel(r"$t$",fontsize=fontsize)
        axes[2][1+transient].set_xlabel(r"$\omega$",fontsize=fontsize)
        
        

        # saving the panel
        
        specific_name+=f"_node{node}"
        svg_path=save_figure(fig, save_dir=save_dir, name=specific_name+".svg")
        png_path=os.path.join(save_dir,specific_name+".png")
        svg2png(url=svg_path,write_to=png_path,
                    parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)
        
        #vtn_scenario_powers(t_final,scenario,model,absolute=False,save_dir=save_dir,epsilon=amplitude,specific_name=power_plot_name,fontsize=fontsize)


def probe_network(model, 
                  shift_matrix_obj=None, 
                  pert_node=1,
                  pert_amplitude=0.1, 
                  num=100,t_final=800, 
                  steps=16000, node=5, 
                  save_dir=None, 
                  name="large_network", 
                  overwrite=True ,
            onset_t=0,
            onset_steady=500,
            t_window=30, 
            plot_real_psd=True,
            linewidth=1.,
            euler_maruyama=False):
    """
    Creates a plot of the network and two additional panels underneath side by side with the following content: 
    steady state response to white continuous noise and psd of the steady state response and theoretical psd on top 
    """
    # create subfigure grid
    fig = plt.figure(figsize=(3, 2), layout='constrained')
    axs = fig.subplot_mosaic([["network", "network"],
                          ["trajectory_pert", "psd_pert"],
                          ["trajectory", "psd"]],)
    
    # create network plot
    plot_network(
            model,
            seed=42,
            #vtn_edge_style="dashed",
            #vtn_color=scenario_color,
            name="network.svg", 
            fs=10,
            threshhold=1e-10,
            perturbed_node=None,
            node_colors="black",
            #axes=axes[0],
            axes=axs["network"],
            node_size=10,
            labels=False
        )
    
    vtn_on=True if shift_matrix_obj!=None else False
    print("vtn_on:", vtn_on)
    # create the scenario
    print("Creating scenario.")
    all_orig_freqs = model.predict_resonance_frequencies()
    all_orig_freqs = all_orig_freqs[np.invert(np.isnan(all_orig_freqs))]
    pert_freqs=np.linspace(np.min(all_orig_freqs),np.max(all_orig_freqs),num=num)
    #pert_freqs=np.linspace(np.max(all_orig_freqs)+1,np.max(all_orig_freqs)+3,num=num)
    #pert_freqs=np.array([1,2,3,4,5.,6.,7,8,9,10,])
    #amplitudes=pert_amplitude*pert_freqs**(-5/3)
    amplitudes=np.ones_like(pert_freqs)*pert_amplitude/len(pert_freqs)
    pert = (dynamics.sine_perturbation_single_node, (amplitudes,pert_freqs,pert_node,onset_t))

    if vtn_on:
        shift_args = (shift_matrix_obj, model, None)
        shift = (dynamics.jacobian_shift, shift_args)
    else:
        shift=(None,None)    
    scenario=pert+shift

    

    network_w_response(scenario, 
                            model, 
                            t_final, 
                            axes=[axs["trajectory_pert"],axs["psd_pert"]],  
                            color=np.array(red),
                            only_perturbation=True, 
                            y_0=None, 
                            steps=steps, 
                            scenario_name=name, 
                            overwrite=overwrite, 
                            vtn_on=False, 
                            node=node, 
                            t_window=t_window,
                            plot_real_psd=plot_real_psd,
                            labels=False,
                            save_dir=save_dir,
                            panel="b",
                            cmap="coolwarm",
                            linewidth=linewidth,
            euler_maruyama=euler_maruyama,
            onset_steady=onset_steady)
    
    network_w_response(scenario, 
                                model, 
                                t_final, 
                                axes=[axs["trajectory"],axs["psd"]],   
                                color="black",
                                y_0=None, 
                                steps=steps, 
                                scenario_name=name, 
                                overwrite=False, 
                                vtn_on=False, 
                                node=node, 
                                t_window=t_window,
                                min_max=None,
                                plot_real_psd=plot_real_psd,
                                labels=False,
                        save_dir=save_dir,
                        panel="c",
                        cmap="coolwarm",
                        linewidth=linewidth,
            euler_maruyama=euler_maruyama,
            onset_steady=onset_steady)

    if False:
        # fetch trajectory and plot
        print("The simulation bit.")
        traj=generate_or_fetch_scenario_data(t_final, scenario, model, steps=steps, meta_scenario_name=name, overwrite=overwrite, minus_fixpoint=True, silent=False)
        t=traj["t"]
        if vtn_on:
            vals=traj["ys_shifted"][:np.shape(model.jacobian_matrix)[0],:] #fetching only the node trajectories, not the velocity trajectories
        else:
            vals=traj["ys"][:np.shape(model.jacobian_matrix)[0],:]

        #axs["trajectory"].plot(t[-500:],vals[node,-500:])
        pert_vals=np.zeros((np.shape(model.jacobian_matrix)[0],steps+1))
        pert_vals=dynamics.sine_perturbation_single_node(t,pert_vals[:,0],scenario[1])
        print("plotting trajectory")

        t_index= np.abs(t - t_window).argmin()
        #axs["trajectory"].plot(t[:t_index],vals[pert_node,:t_index],alpha=0.7)
        axs["trajectory"].plot(t[:t_index],pert_vals[pert_node,:t_index],color="red",linewidth=1.,alpha=0.7)
        
        # calculate and plot psd
        print(len(t), len(vals[pert_node]))
        t=t[20000:]
        node_vals=vals[pert_node,20000:]
        print(len(t), len(node_vals))
        
        print("plotting and calculating psd")
        freqs, psd = scipy.signal.welch(node_vals, fs=1/(t[1]-t[0]), axis=0, nperseg=len(t))
        axs["psd"].plot(2*np.pi*freqs,1/pert_amplitude*psd)
        axs["psd"].set_yscale("log")
        #axs["psd"].set_ylim(pert_amplitude/num*1e-2,1.5*np.max(psd))
        axs["psd"].set_ylim(1e-5,5*np.max(psd)/pert_amplitude)
        axs["psd"].set_xlim(0,10)

        # calculate theoretical response amplitudes and plot it on top as a thin black line 
        response_vals=model.calculate_response_amplitudes(freqs,k=pert_node)
        for i in range(np.shape(model.jacobian_matrix)[0]):
            axs["psd"].plot(freqs,response_vals[i,:],color="black" if i==node else "green",linewidth=1., alpha=0.1 if i!=node else 1.)
    #plt.show()

    # save the figure
    if save_dir== None and shift_matrix_obj==None:
        save_dir= model.current_dir
    elif save_dir== None:
            save_dir = shift_matrix_obj.current_dir
    name+="_num_of_modes_"+str(num)
    if euler_maruyama:
        name+="_euler_maruyama"
    svg_path=save_figure(fig, save_dir=save_dir, name=name+".svg")
    png_path=os.path.join(save_dir,name+".png")
    svg2png(url=svg_path,write_to=png_path,
                    parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)


if __name__ == "__main__":
    
    # Create a model and compute the Jacobian
    #model = sokm.illustrative_8node()
    #model = sokm.from_random_sparse_graph(num_nodes=8, edge_probability=0.1, damping_coefficient=0.01)
    #model = sokm.from_soft_random_geometric_graph(num_nodes=8, radius=0.3, damping_coefficient=0.01)
    #model.compute_jacobian()
    #model.summary()
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-05-11_17-20-36")
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-06-02_16-36-00")
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-06-03_15-42-09")
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-06-11_12-56-31")
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\8node")
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Moritz_vals")
    model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\Example network")
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\moritz_a_001")
    #model.compute_jacobian()
    

    # Generate a shift matrix
    shift_matrix_obj = ShiftMatrix(model=model)

    eigenvalue_indices = [ 2,5]
    shifts = np.array([-2.5,-1.5],dtype=float)
    #shifts = np.array([-2.5,-2.5],dtype=float)
    #zero_rows = np.array([5,6,7],dtype=int)
    #zero_rows=np.arange(5, dtype=int)
    #zero_rows=np.array([1,2,3,4,5])
    #zero_rows=np.arange(np.shape(model.jacobian_matrix)[0]-len(eigenvalue_indices)-73, dtype=int)
    zero_rows=shift_matrix_obj.ideal_zero_rows( num_vtn_nodes=3, eigenvalue_indices=eigenvalue_indices)
    zero_cols = np.array([],dtype=int)
    shift_matrix_obj.construct_from_scratch(eigenvalue_indices, shifts, zero_rows, zero_cols)
    #shift_matrix_obj= ShiftMatrix.load_from_file("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-03-31_10-32-17\\shift_matrix.json")
    
    print(f"Xs: {shift_matrix_obj.Xs} and Ys: {shift_matrix_obj.Ys}")

    #compose_shift_matrix_construction_visualization(shift_matrix_obj, log=True, absolute=True,fs=8, jac_color=black)
    perturbed_node=5
    t_final=1000
    steps=16000
    meta_scenario_name="sine_scenario"
    #angle_comparison(model, shift_matrix_obj)
    nodes=[0,3]
    #generate_scenarios(model, noise_type="realistic", perturbation_strength=0.1, frequency_samples=400 , steps=8000)
    if False:
        plot_dynamics_evaluation(t_final,
                                    model,
                                    nodes=nodes,
                                    perturbation_strength=0.01,
                                    steps=steps,
                                    y_0=None,
                                    name="dynamics_realistic", 
                                    save_dir=None,
                                    fontsize=5,
                                    perturbed_node=perturbed_node,
                                    noise_type="realistic",
                                    overwrite=False
                                    )
        
    probe_network(model, overwrite=True, pert_amplitude=0.1, num =100, steps=steps*20, t_final= t_final, euler_maruyama=True, name="white_noise",onset_steady=750)

    if False:
        plot_network(
                    model,
                    seed=42,
                    name="network", 
                    fs=10,
                    threshhold=1e-10,
                    node_colors=[0.,0.,0.,0.],
                    perturbed_node=None,
                    node_size=50,
                    labels=False,
                    show_node_type=True
                )
            #scenarios = generate_scenarios(model, noise_type="sine", perturbation_strength=0.1, frequency_samples=400 , steps=steps, overwrite=True)
            #illustrative(model, amplitude=0.1, overwrite=False, node=4, plot_real_psd=False, labels=False)
        print(type(shift_matrix_obj))
        scenario_panel_recursive(model,
                                    shift_matrix_obj,
                                    t_final,
                                    steps=steps,
                                    color=orange, 
                                    save_dir=None, 
                                    specific_name="scenario_panel", 
                                    overwrite=False,
                                    node=4, 
                                    t_window=10,
                                    pert_node=perturbed_node,
                                    labels=False,
                                    fontsize=10,
                                    plot_real_psd=False,
                                    onset_t=2
                                    )
        # noise or driving, what is the difference?
        # ways forward: 
        # write integrator yourself ( but if I mess it up, that will be very shitty)
        # move to Julia
        # normal Ode integration (but then adding noise AFTER integration)
        # continuous perturbation (not using fft but summing over sines with frequencies and random phases)
        