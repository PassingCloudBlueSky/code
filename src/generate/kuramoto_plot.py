from platform import node

from matplotlib.pylab import False_, eig

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


def resonance_plot(model: sokm,
                   shift_matrix_obj: Optional[ShiftMatrix]=None,
                   name="resonance_plot.svg",
                   omega=None, 
                   pert_band=None,
                   min_max=None,
                   log=False,
                   extra_scope=0.2,
                   show_resonance_location=False,
                   fs=6,
                   lw=1,
                   alpha=0.8,
                   x_axis_off=False,
                   perturbed_node=6,
                   noise_object=None):
    """
    Creates a resonance plot for a network with Jacobian J and a perturbation at node k. 
    Plots the frequency-dependent response amplitudes for all nodes in the network over a 
    specified frequency range. Optionally displays resonance frequency peaks and perturbation bands.
                Parameters
                ----------
                model : sokm
                    The network model containing the second order Kuramoto model Jacobian and system dynamics.
                shift_matrix_obj : Optional[ShiftMatrix], default=None
                    Optional shift matrix object for eigenvalue manipulation. If provided, shifted resonance
                    frequencies are displayed. Those shifted with corresponding colors.
                name : str, default="resonance_plot.svg"
                    Filename for the saved plot.
                omega : array-like, optional
                    Frequency array for the x-axis. If None, automatically generated based on the model's
                    resonant behavior with extra_scope padding.
                pert_band : tuple[float, float], optional
                    Perturbation band region as (start_frequency, width) to highlight in the plot.
                min_max : tuple[float, float], optional
                    Y-axis limits as (min_value, max_value). If None, limits are set automatically.
                log : bool, default=False
                    If True, use logarithmic scale for the y-axis.
                extra_scope : float, default=0.2
                    Extra frequency range padding (relative to resonant region) when auto-generating omega.
                show_resonance_location : bool, default=False
                    If True, display dashed vertical lines at resonance frequency peaks. Colors indicate
                    whether peaks correspond to shifted eigenvalues.
                fs : int, default=6
                    Fontsize for axis labels and tick labels.
                lw : float, default=1
                    Linewidth for response amplitude curves.
                alpha : float, default=0.8
                    Transparency level for response amplitude curves.
                x_axis_off : bool, default=False
                    If True, hide x-axis ticks and labels.
                perturbed_node : int, default=6
                    Index of the perturbed node for which response is calculated.
                Returns
                -------
                str
                    Path to the saved plot file.
    """    
    
    # generate frequency array omega in the area of resonant behavior unless provided as parameter
    if omega==None:
        omega=generate_frequency_range(model,extra_scope=extra_scope)

    # fetch shift matrix
    if shift_matrix_obj is not None:
        S=shift_matrix_obj.shift_matrix
    else:
        S=None

    response_vals=model.calculate_response_amplitudes(omega,S=S,k=perturbed_node)
    

    # loop over all nodes, and plot their freuquency dependent response amplitudes into one graph
    fig,ax=plt.subplots(1, 1,figsize=(2.1, 0.7))
    psd_color="gold"
    response_color=green

    # plot power spectral densities of the perturbation if provided as parameter
    if noise_object is not None:
        twin_ax=ax.twinx()
        for i in range(noise_object.psd.ndim):
            twin_ax.plot(2*np.pi*noise_object.freqs,noise_object.psd[i,:] if noise_object.psd.ndim > 1 else noise_object.psd,"o",markersize=0.5,alpha=1.,color=psd_color,linewidth=lw*0.5)
        
    # option to visualize the resonance frequency peaks with dashed vertical lines
    if show_resonance_location:

        # fetching colors of individual shifts if shifted
        if shift_matrix_obj!=None:
            shift_colors=create_shift_colors(len(shift_matrix_obj.eigenvalue_indices))
        
        # locations of resonance peaks
        resonance_frequencies=model.predict_resonance_frequencies(S=S)

        # looping over all peaks to add the vertical line to the plot
        for i, current_frequency in enumerate(resonance_frequencies):
            line_color="grey"
            v_alpha=0.5

            # checking if shifted
            if shift_matrix_obj!=None:
                
                # calculating the predictions for the shifted resonance frequencies
                shifted_eigvals=shift_matrix_obj.eigenvalues[shift_matrix_obj.eigenvalue_indices]+shift_matrix_obj.shifts
                shifted_resonance_frequencies=model.predict_resonance_frequencies(shifted_eigvals=shifted_eigvals)

                # creating a boolean array with as many entries as shifts. A True entry means that the current resonance frequency is in 1e-8 proximity to the predicted resoannce frequency of the shift
                rel_diffs = np.abs(shifted_resonance_frequencies - current_frequency) / np.abs(current_frequency)
                in_tol = rel_diffs < 1e-8

                # if the current resonance frequency corresponds to any predicted resonance frequency
                if np.any(in_tol):
                    # set the line color  to the shift color of that resonance frequency
                    line_color=shift_colors[np.where(in_tol)[0][0]]
                    v_alpha=1

            # draw vertical line
            ax.axvline(x=current_frequency,linestyle="dashed",color=line_color,alpha=v_alpha,linewidth=1.)

    
    # Drawing perturbation band if provided
    if pert_band!=None and noise_object is None: 
        rect=plt.Rectangle((pert_band[0],0), pert_band[1], 20*np.max(response_vals), color=psd_color,alpha=0.3,edgecolor=None)
        ax.add_patch(rect)

    # loop over all nodes, and plot their freuquency dependent response amplitudes into one graph
    for i in range(len(response_vals[:,0])): #loop over all nodes
        ax.plot(omega,response_vals[i,:],alpha=alpha,color=response_color,linewidth=lw)

    # log or not log that is the question
    if log: 
        #plt.xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8*np.min(response_vals))
        ax.set_ylabel("$A_n$",fontsize=fs)
        if noise_object is not None:
            twin_ax.set_yscale("log")
            twin_ax.set_ylim(bottom=0.8*np.min(noise_object.psd))
            twin_ax.set_ylabel("PSD",fontsize=fs,color=psd_color)
            ax.set_ylabel("$A_n$",fontsize=fs,color=response_color,alpha=1.)
    else:
        ax.set_ylim(bottom=0)
        ax.set_ylabel("$A_n$",fontsize=fs)

    if x_axis_off:
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        if noise_object is not None:
            twin_ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) 
    else:
        ax.set_xlabel("$\omega$",fontsize=fs)

    # aesthetics
    ax.set_xlim((omega[0],omega[-1]))
    if min_max!=None:
        ax.set_ylim(min_max)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    if noise_object is not None:
        twin_ax.tick_params(axis='y', labelsize=fs, color=psd_color,labelcolor=psd_color)
        response_color=response_color[:3]
        ax.tick_params(axis='y', labelcolor=response_color)
    for axis in ['top','bottom','left','right']: #thicker axis
        ax.spines[axis].set_linewidth(0.6)
    ax.set_facecolor("white")

    # ascertaining existence of model instance directory
    if model.current_dir == None:
        model.save_parameters()

    if shift_matrix_obj!=None:
        save_dir=shift_matrix_obj.current_dir
    else:
        save_dir = model.current_dir
    save_path=save_figure(fig, save_dir, name=name)
    return save_path

    
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



def perturbation_band_location_from_shift(model:sokm, shift_matrix_obj:ShiftMatrix, buffer=0.05):
    """
    Reverse engineers the left edge and width of a perturbation band that justifies shifting the eigenvalues with shift_matrix_obj.

    Parameters
    -------
    model: sokm
        Second Order Kuramoto Model instance from which the resonant frequencies are collected.
    shift_matrix_obj: ShiftMatrix
        ShiftMatrix instance in which a shift applied to model is stored. 
    buffer:
        Closest distance between perturbation band edges and the predicted resonance frequencies inside the perturbation band. 

    Returns
    ----------
    tuple[float, float]
        Tuple of the left edge value of the perturbation band and the perturbation band's width
    """


    # fetch shift indices and determine edge indices
    shift_indices=shift_matrix_obj.eigenvalue_indices
    min = np.min(shift_indices)
    max= np.max(shift_indices)

    # fetch corresponding predicted resonance frequencies to calculate left edge of perturbation band its width. Extend with buffer on both sides 
    resonance_frequencies=model.predict_resonance_frequencies()
    left_edge= resonance_frequencies[max] - buffer
    width = resonance_frequencies[min]-left_edge + buffer
    
    return left_edge,width


def pre_and_post_shift_comparison_subplots(model:sokm, shift_matrix_obj:ShiftMatrix, log=True, threshhold=1e-10,fs=5,perturbed_node=6, noise_object=None, edge_weight_visualization="color", node_colors=None):

    """
    Generate individual subplots comparing network response before and after applying eigenvalue shifts.

    Creates four plots showing resonance response amplitudes and network topology with and without
    the virtual transmission network (VTN). Returns paths to all generated plot files for composition.

    Parameters
    ----------
    model : sokm
        The network model containing the second order Kuramoto model Jacobian and system dynamics.
    shift_matrix_obj : ShiftMatrix
        ShiftMatrix instance containing eigenvalue shifts to be applied to the model.
    log : bool, default=True
        If True, use logarithmic scale for response amplitude y-axis.
    threshhold : float, default=1e-10
        Threshold for identifying non-zero entries in the shift matrix rows that indicate
        nodes requiring active control in the network visualization; are understood as VTN nodes.
    fs : int, default=5
        Fontsize for axis labels and tick labels in all subplots.
    perturbed_node : int, default=6
        Index of the perturbed node for which the response is calculated.

    Returns
    -------
    list[str]
        List of paths to the saved plot files in order: [unshifted_resonance, shifted_resonance, 
        unshifted_network, shifted_network].
    """

    
    paths=[]
    
    # calculate perturbation band rectangle left and right x values 
    pert_band = perturbation_band_location_from_shift(model, shift_matrix_obj)
    
    # construct omega array
    omega=generate_frequency_range(model)

    # calculate y min and max values 
    response_vals_unshifted=model.calculate_response_amplitudes(omega)
    response_vals_shifted=model.calculate_response_amplitudes(omega,S=shift_matrix_obj.shift_matrix)
    all_response_vals=np.vstack((response_vals_shifted,response_vals_unshifted))
    min_max=(np.min(all_response_vals)*1.1,1.1*np.max(all_response_vals))
    #if noise_object is not None:
        #min_max=(min(min_max[0], 0.001*np.max(noise_object.psd)), max(min_max[1], 1.1*np.max(noise_object.psd)))

    # fetch unshifted resonance plot data & plot
    print("Visualizing network without VTN...")
    paths.append(resonance_plot(model, pert_band=pert_band, min_max=min_max, name="res.SVG", log=log, extra_scope=0.2, show_resonance_location=True, fs=fs,lw=1,alpha=1.,x_axis_off=True,perturbed_node=perturbed_node,noise_object=noise_object))
    # generate shifted resonance data and plot
    print("Visualizing network with VTN...")
    paths.append(resonance_plot(model, shift_matrix_obj=shift_matrix_obj, pert_band=pert_band, min_max=min_max, name= "res_shifted.SVG",log=log, extra_scope=0.2, show_resonance_location=True, fs=fs,lw=1,alpha=1.,perturbed_node=perturbed_node,noise_object=noise_object))

    # generate network plot WITH seed
    print("Plotting unshifted response amplitudes...")
    paths.append(plot_network( model, seed=4, name="network.SVG", edge_weight_key="weight", edge_weight_visualization=edge_weight_visualization, perturbed_node=perturbed_node, threshhold=threshhold, node_colors=node_colors))

    # generate network plot with VTN connecting nodes corresponding to non zero rows of the shift matrix
    print("Plotting shifted response amplitudes...")
    paths.append(plot_network( model, shift_matrix_obj=shift_matrix_obj, seed=4, name="network_VTN.SVG", edge_weight_visualization=edge_weight_visualization,  perturbed_node=perturbed_node, threshhold=threshhold, node_colors=node_colors ))
    
    return paths


def compose_pre_and_post_shift_comparison(model:sokm, shift_matrix_obj:ShiftMatrix, log=True, vtn_threshhold=1e-10, fontsize=5, perturbed_node=6, noise_object=None, name="comparison", edge_weight_visualization="color",node_colors=None):

    """
    Generate individual subplots comparing network response before and after applying eigenvalue shifts.

    Composes the four plots showing resonance response amplitudes and network topology with and without
    virtual tunable nodes (VTN) into one figure.

    Parameters
    ----------
    model : sokm
        The network model containing the second order Kuramoto model Jacobian and system dynamics.
    shift_matrix_obj : ShiftMatrix
        ShiftMatrix instance containing eigenvalue shifts to be applied to the model.
    log : bool, default=True
        If True, use logarithmic scale for response amplitude y-axis.
    vtn_threshhold : float, default=1e-10
        Threshold for identifying non-zero entries in the shift matrix rows that indicate
        nodes requiring active control in the network visualization.
    fs : int, default=5
        Fontsize for axis labels and tick labels in all subplots.
    perturbed_node : int, default=6
        Index of the perturbed node for which the responses are calculated.
    freqs_psds : tuple of array-like, optional
        A tuple containing frequency values and corresponding power spectral density values to be plotted.

    Returns
    -------

    """


    print("Composing network visualizations and response amplitude plots...")
    paths=pre_and_post_shift_comparison_subplots(model, shift_matrix_obj, log=log, threshhold=vtn_threshhold,fs=fontsize, perturbed_node=perturbed_node, noise_object=noise_object, edge_weight_visualization=edge_weight_visualization, node_colors=node_colors)

    fig = sg.SVGFigure("7in", "2.2in")
    #fig.append(sc.Grid(10,10)) # visual grid for ease of aligning figures

    # vertically stacking response amplitude plots
    x_0=110
    y_0=45
    for i in range(2):
        fig_part = sg.fromfile(paths[i])
        res_plot = fig_part.getroot()
        res_plot.moveto(x_0, y_0*i, scale_x=1, scale_y=1)
        reference=sg.TextElement(x_0+5,i*y_0+5, chr(ord('`')+i+2)+")", size=6)
        fig.append([res_plot,reference])

    # load and place the network plots to the left and right respectively
    noise_offset=20 if noise_object is not None else 0 
    for i in range(2,4):
        fig_part = sg.fromfile(paths[i])
        network_plot = fig_part.getroot()
        network_plot.moveto((i-2)*(260+noise_offset), 0, scale_x=1, scale_y=1)
        if i==2:
            reference=sg.TextElement((i-2)*260+5,5, "a)", size=6)
        else:               
            reference=sg.TextElement((i-2)*(260+5+noise_offset),5, "d)", size=6)
        fig.append([network_plot,reference])

        
    # saving as SVG and PNG
    save_dir=os.path.join(shift_matrix_obj.current_dir,name+".svg")
    fig.save(save_dir)
    png_path=os.path.join(shift_matrix_obj.current_dir,name+".png")
    svg2png(url=save_dir,write_to=png_path,
            parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)


def generate_dynamics_comparison(t_final,
                               model:sokm,
                               shift_matrix_obj:ShiftMatrix,
                               perturbation_strength=0.05,
                               steps=8000,
                               y_0=None,
                               name="dynamics_comparison", 
                               save_dir=None,
                               fontsize=5,
                               perturbed_node=6,
                               noise_type="white",
                               frequency_samples=400,
                               nodes=[0,3,5]):
    
    
    # making sure that input is valid
    if noise_type not in ("exp_gaussian", "gaussian", "white", "realistic", "realistic+peak"):
        raise ValueError("noise_type must be 'exp_gaussian' or 'gaussian' or 'white' or 'realistic' or 'realistic+peak'")

    # Calculating the resonance frequencies affected by the shift prior to the shift
    all_orig_freqs = model.predict_resonance_frequencies()
    all_orig_freqs = all_orig_freqs[np.invert(np.isnan(all_orig_freqs))]
    prior_shift_freqs=all_orig_freqs[shift_matrix_obj.eigenvalue_indices]

    # checking if step size is sufficient to capture the non-local dynamics of the network
    if np.max(all_orig_freqs)*10>steps/t_final:
        print("WARNING: The chosen step size might be too large to capture the non-local dynamics of the network. Consider increasing the number of steps or decreasing t_final for a more accurate representation of the dynamics.")
    print(f"Time scale of largest eigenvalue is {1/np.max(all_orig_freqs)} while the time step size is {t_final/steps}. Consider adjusting these parameters if the time step size is not significantly smaller than the time scale of the largest eigenvalue for a more accurate representation of the dynamics.")

    #creating scenarios
    scenarios=[]

    # creating offset only scenario
    #scenarios.append((r"initial offset", (None,5*perturbation_strength)))

    # creating noise from psd scneario
    left, width = perturbation_band_location_from_shift(model, shift_matrix_obj, buffer=0.05)
    width = width/(2*np.pi) # 
    left = left/(2*np.pi)
    center = left+width/2

    if noise_type=="exp_gaussian":
        exp_gaussian_params = {'max_ampl': perturbation_strength, 'cutoff': np.max(all_orig_freqs)/(2*np.pi), 'center': center, 'width': width*0.3/2}
        noise_object = dynamics.ContinuousSpectrumNoise(dynamics.exp_gaussian_spectrum, exp_gaussian_params, max_val=0.05, steps=frequency_samples, rate=1.)
        noise_label=r"exp-gaussian"
    elif noise_type=="gaussian":
        gaussian_params = {'max_ampl': perturbation_strength, 'center': center, 'width': width*0.3/2}
        noise_object = dynamics.ContinuousSpectrumNoise(dynamics.gaussian_spectrum, gaussian_params, max_val=0.05, steps=frequency_samples, rate=1.)
        noise_label=r"gaussian"
    elif noise_type=="white":
        white_noise_params = {'max_ampl': perturbation_strength, 'freq_min': left, 'freq_max': left + width}
        noise_object = dynamics.ContinuousSpectrumNoise(dynamics.white_spectrum, white_noise_params, max_val=0.05, steps=frequency_samples, rate=1.)
        noise_label=r"white"
    elif noise_type=="realistic":
        realistic_params = {'cutoff': np.max(all_orig_freqs)/(2*np.pi), 'min': 0.05}
        noise_object = dynamics.ContinuousSpectrumNoise(dynamics.realistic_spectrum, realistic_params, max_val=0.05, steps=frequency_samples, rate=1.)
        noise_label=r"realistic"
    elif noise_type=="realistic+peak":
        realistic_peak_params = {'cutoff': np.max(all_orig_freqs)/(2*np.pi), 'min': 0.05, 'center': center, 'width': width*0.3/2}
        noise_object = dynamics.ContinuousSpectrumNoise(dynamics.realistic_spectrum_with_peak, realistic_peak_params, max_val=0.05, steps=frequency_samples, rate=1.)
        noise_label=r"realistic + peak"

    path=noise_object.save(shift_matrix_obj.current_dir)
    #print(path)
    #noise_object=noise_object.from_file(path)
    plot_noise(noise_object, shift_matrix_obj, name=noise_label, save_dir=None)
    pert = (dynamics.perturbation_from_psd, (noise_object,perturbed_node))
    scenarios.append((noise_label, pert))
    
    # creating color coding of nodes according to their distance to the perturbed node for the network plot
    cmap=plt.cm.cividis.reversed()
    node_colors = node_colors_from_perturbation_distance(perturbed_node, model, cmap=cmap)

    # adding the power spectral densities to the response amplitude plot
    compose_pre_and_post_shift_comparison(model, shift_matrix_obj, log=True, vtn_threshhold=1e-10, fontsize=fontsize, perturbed_node=perturbed_node, noise_object=noise_object, name=noise_type+"_comparison", node_colors=node_colors, edge_weight_visualization="color")

    # creating resonance scenarios
    
    for i in np.invert(range(0, len(prior_shift_freqs))):
        args_sine = (perturbation_strength, prior_shift_freqs[i] , perturbed_node)
        scenarios.append(
            (r"$\omega=$"+f"{np.round(prior_shift_freqs[i], 3)}", (dynamics.sine_perturbation_single_node, args_sine))
        )

    # forwarding the scenarios to the plot_utils function
    if False:
        return plot_scenario_comparison(t_final,
                               model,
                               shift_matrix_obj,
                               scenarios,
                               steps=steps,
                               y_0=y_0,
                               name= noise_type+"_"+name, 
                               save_dir=save_dir,
                               fontsize=fontsize+5,
                               perturbed_node=perturbed_node)

    plot_scenario_comparison_split(t_final,
                               model,
                               shift_matrix_obj,
                               scenarios,
                               nodes=nodes,
                               steps=8000,
                               y_0=y_0,
                               name="dynamics_comparison", 
                               save_dir=None,
                               fontsize=5,
                               perturbed_node=1)


def plot_dynamics_evaluation(t_final,
                            model:sokm,
                                load_scenarios=False,
                               perturbation_strength=0.05,
                               steps=8000,
                               name="dynamics_evaluation", 
                               save_dir=None,
                               y_0=None,
                               fontsize=10,
                               perturbed_node=6,
                               noise_type="white",
                               frequency_samples=400,
                               nodes=[0,3],
                               overwrite=False,
                               alpha=1.):
    """
    Creates a publication ready plot comparing two different VTN setup fro the same BESS setup based on BESS powers (as a func of time and averaged), 
    individual node trajectories, power spectral density of the response. The trajectories and PSD are inside the subplot compared to the response without VTN.
    On top of the two resulting rows are a plot of the network and one of of the driving signal psd. Inside the psd Plots the vtn affected response frequencies
    are annotated via vlines. 
    """
    
    if y_0 is None:
        y_0=np.zeros(model.ode_dimension)
        y_0[:model.jacobian_matrix.shape[0]] = model.compute_fixed_point()

    # generate scenarios:
    scenarios=generate_scenarios(model, noise_type=noise_type, perturbation_strength=perturbation_strength, frequency_samples=frequency_samples,steps=steps,overwrite=overwrite) if not load_scenarios else load_scenarios

    # create the axes for the figure
    ONE_MM = 1 / 25.4  # Convert mm to inches
    fig = plt.figure(figsize=(85*ONE_MM*2,70*ONE_MM))
    w=6
    h=2
    
    grid=(3*h,3*w)
    ax_network=plt.subplot2grid(grid, (0,0), rowspan=h, colspan=2*w, fig=fig)
    ax_psd=plt.subplot2grid(grid, (0,2*w), rowspan=h, colspan=w, fig=fig)
    scenario_axes={}
    for i in range(1,len(scenarios)+1):
        axes = []
        axes.append(plt.subplot2grid(grid, (h*i,0), rowspan=h, colspan=w, fig=fig))
        axes.append(plt.subplot2grid(grid, (h*i,w), rowspan=int(h/2), colspan=w, fig=fig))
        axes.append(plt.subplot2grid(grid, (h*i+1,w), rowspan=int(h/2), colspan=w, fig=fig))
        axes.append(plt.subplot2grid(grid, (h*i,2*w), rowspan=h, colspan=w, fig=fig))
        scenario_axes[f"ax_row{i}"]=axes

    
    # plot  network
    shift_matrix_obj= scenarios[0][3][0]
    plot_network( model, shift_matrix_obj=shift_matrix_obj, axes=ax_network, seed=4, name="network_VTN.SVG",  perturbed_node=perturbed_node, node_colors="black" )

    # calculate vline location of resonances shifted w and w/o VTN

    # plot perturbation psd histogram 
    noise_object = scenarios[0][1][0] #fetching the noise object from the first scenario, which is the only one with noise in our current setup
    freqs=noise_object.freqs/2*np.pi
    noise_psd=noise_object.psd
    filter=noise_psd>0
    freqs=freqs[filter]
    noise_psd=noise_psd[filter]
    freqs_lims=(freqs[0], freqs[-1])
    min_max_psd = plot_psd(ax_psd,freqs,noise_psd,freqs_vtn=None,psd_vtn=None,freqs_lims=freqs_lims)

    # as a separate function: plot scenario row (generate data or not based on load_scenarios parameter)
    for i,scenario in enumerate(scenarios):
        psd_vlines, psd_vline_colors = v_line_locations_from_shift(model, scenario[3][0])
        min_max_psd = plot_scenario_row(t_final, 
                                        scenario, 
                                        model, 
                                        y_0, 
                                        scenario_axes[f"ax_row{i+1}"], 
                                        nodes=nodes, 
                                        steps=steps, 
                                        meta_scenario_name=f"scenario{i+1}",
                                        overwrite=overwrite,
                                        freqs_lims=freqs_lims, 
                                        min_max_psd=min_max_psd,
                                        psd_vlines= psd_vlines,
                                        psd_vline_colors=psd_vline_colors,
                                        alpha=alpha)
    
    min_max_psd[0] = max(min_max_psd[0],1e-8)
    ax_psd.set_ylim(min_max_psd)
    for i in range(1,len(scenarios)+1):
        scenario_axes[f"ax_row{i}"][3].set_ylim(min_max_psd)
    

    if save_dir is None:
        save_dir = model.current_dir
    svg_path=save_figure(fig, save_dir=save_dir, name=name+"_tfinal"+str(t_final)+".svg")
    png_path=os.path.join(save_dir,name+"_tfinal"+str(t_final)+".png")
    svg2png(url=svg_path,write_to=png_path,
                parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)
    return svg_path


def generate_scenarios(model:sokm, noise_type="realistic", perturbation_strength=0.1, frequency_samples=400 , steps=8000, overwrite=True):
    """
    A helper function to generate the the scenarios for handling in plot dynamics evaluation and ultimately dynamics.integrate_f.
    THe hardcoded scenarios are:
        - 3 BESS in identical loactions
        - one shift and two shifts
        - identical driving of the system
    """

    # making sure that input is valid
    if noise_type not in ("sine","exp_gaussian", "gaussian", "white", "realistic", "realistic+peak"):

        raise ValueError("noise_type must be 'sine','exp_gaussian' or 'gaussian' or 'white' or 'realistic' or 'realistic+peak'")


    # checking if step size is sufficient to capture the non-local dynamics of the network
    all_orig_freqs = model.predict_resonance_frequencies()
    all_orig_freqs = all_orig_freqs[np.invert(np.isnan(all_orig_freqs))]

    if np.max(all_orig_freqs)*10>steps/t_final:
        print("WARNING: The chosen step size might be too large to capture the non-local dynamics of the network. Consider increasing the number of steps or decreasing t_final for a more accurate representation of the dynamics.")
    print(f"Time scale of largest eigenvalue is {1/np.max(all_orig_freqs)} while the time step size is {t_final/steps}. Consider adjusting these parameters if the time step size is not significantly smaller than the time scale of the largest eigenvalue for a more accurate representation of the dynamics.")


    shift_matrix_obj_a = ShiftMatrix(model=model)
    shift_matrix_obj_b = ShiftMatrix(model=model)

    # setting vtn parameters and calculating rudimentary ideal vtn node locations for the given shifts and vtn size
    #zero_rows = np.array([5,6,7],dtype=int)
    num_vtn_nodes = 3
    zero_cols = np.array([],dtype=int)
    eigenvalue_indices_a = [ 2,3]
    shifts_a = np.array([-0.6,0.3],dtype=float)
    eigenvalue_indices_b = [ 5]
    shifts_b = np.array([-0.5],dtype=float)
    zero_rows = shift_matrix_obj_a.ideal_zero_rows(num_vtn_nodes, eigenvalue_indices_a+eigenvalue_indices_b) 

    # generate shift matrices    
    shift_matrix_obj_a.construct_from_scratch(eigenvalue_indices_a, shifts_a, zero_rows, zero_cols)
    shift_args_a = (shift_matrix_obj_a, model, None)
    shift_a = (dynamics.jacobian_shift, shift_args_a)
    compose_shift_matrix_construction_visualization(shift_matrix_obj_a, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=overwrite)

    
    shift_matrix_obj_b.construct_from_scratch(eigenvalue_indices_b, shifts_b, zero_rows, zero_cols)
    shift_args_b = (shift_matrix_obj_b, model, None)
    shift_b = (dynamics.jacobian_shift, shift_args_b)
    compose_shift_matrix_construction_visualization(shift_matrix_obj_b, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=overwrite)
    
    #all_orig_freqs = all_orig_freqs[np.invert(np.isnan(all_orig_freqs))]
    #prior_shift_freqs=all_orig_freqs[shift_matrix_obj.eigenvalue_indices]

    
    # generate noise configuration NOTE: currently only "realistic is functional!"
    if noise_type=="exp_gaussian":
        exp_gaussian_params = {'max_ampl': perturbation_strength, 'cutoff': np.max(all_orig_freqs)/(2*np.pi), 'center': center, 'width': width*0.3/2}
        noise_object = dynamics.ContinuousSpectrumNoise(dynamics.exp_gaussian_spectrum, exp_gaussian_params, max_val=0.05, steps=frequency_samples, rate=1.)
        noise_label=r"exp-gaussian"
    elif noise_type=="gaussian":
        gaussian_params = {'max_ampl': perturbation_strength, 'center': center, 'width': width*0.3/2}
        noise_object = dynamics.ContinuousSpectrumNoise(dynamics.gaussian_spectrum, gaussian_params, max_val=0.05, steps=frequency_samples, rate=1.)
        noise_label=r"gaussian"
    elif noise_type=="white":
        white_noise_params = {'max_ampl': perturbation_strength, 'freq_min': left, 'freq_max': left + width}
        noise_object = dynamics.ContinuousSpectrumNoise(dynamics.white_spectrum, white_noise_params, max_val=0.05, steps=frequency_samples, rate=1.)
        noise_label=r"white"
    elif noise_type=="realistic":
        realistic_params = {'cutoff': np.max(all_orig_freqs)/(2*np.pi), 'min': 0.05}
        noise_object = dynamics.ContinuousSpectrumNoise(dynamics.realistic_spectrum, realistic_params, max_val=0.05, steps=frequency_samples, rate=1.)
    elif noise_type=="realistic+peak":
        realistic_peak_params = {'cutoff': np.max(all_orig_freqs)/(2*np.pi), 'min': 0.05, 'center': center, 'width': width*0.3/2}
        noise_object = dynamics.ContinuousSpectrumNoise(dynamics.realistic_spectrum_with_peak, realistic_peak_params, max_val=0.05, steps=frequency_samples, rate=1.)
        noise_label=r"realistic + peak"
    elif noise_type=="sine":

        pert=(dynamics.sine_perturbation_single_node, (amplitude, frequency, perturbed_node))

    noise_object.save(model.current_dir)
    print(f"Saved noise object with {noise_type} spectrum to {model.current_dir}")
    pert = (dynamics.perturbation_from_psd, (noise_object,perturbed_node))
    
    # compose into scenarios 
    return [pert+shift_a,pert+shift_b]

def generate_sine_scenarios(model:sokm, amplitude=0.1, overwrite=True):
    """
    A helper function to generate the the scenarios for handling in plot dynamics evaluation and ultimately dynamics.integrate_f.
    THe hardcoded scenarios are:
        - 3 BESS in identical loactions
        - one shift and two shifts
        - identical driving of the system
    """

    all_orig_freqs = model.predict_resonance_frequencies()
    all_orig_freqs = all_orig_freqs[np.invert(np.isnan(all_orig_freqs))]

    shift_matrix_obj_a = ShiftMatrix(model=model)
    shift_matrix_obj_b = ShiftMatrix(model=model)

    # setting vtn parameters and calculating rudimentary ideal vtn node locations for the given shifts and vtn size
    #zero_rows = np.array([5,6,7],dtype=int)
    num_vtn_nodes = 2
    zero_cols = np.array([],dtype=int)
    eigenvalue_indices_a = [4]
    shifts_a = np.array([-0.6],dtype=float)
    eigenvalue_indices_b = [ 6]
    shifts_b = np.array([-0.5],dtype=float)
    zero_rows = shift_matrix_obj_a.ideal_zero_rows(num_vtn_nodes, eigenvalue_indices_a+eigenvalue_indices_b) 
    # generate shift matrices    
    shift_matrix_obj_a.construct_from_scratch(eigenvalue_indices_a, shifts_a, zero_rows, zero_cols)
    shift_args_a = (shift_matrix_obj_a, model, None)
    shift_a = (dynamics.jacobian_shift, shift_args_a)
    compose_shift_matrix_construction_visualization(shift_matrix_obj_a, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=overwrite)

    
    shift_matrix_obj_b.construct_from_scratch(eigenvalue_indices_b, shifts_b, zero_rows, zero_cols)
    shift_args_b = (shift_matrix_obj_b, model, None)
    shift_b = (dynamics.jacobian_shift, shift_args_b)
    compose_shift_matrix_construction_visualization(shift_matrix_obj_b, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=overwrite)
    
    #all_orig_freqs = all_orig_freqs[np.invert(np.isnan(all_orig_freqs))]
    #prior_shift_freqs=all_orig_freqs[shift_matrix_obj.eigenvalue_indices]

    
    pert_a = (dynamics.sine_perturbation_single_node, (amplitude,all_orig_freqs[eigenvalue_indices_a[0]],perturbed_node))
    
    pert_b = (dynamics.sine_perturbation_single_node, (amplitude,all_orig_freqs[eigenvalue_indices_b[0]],perturbed_node))
    
    # compose into scenarios 
    return [pert_a+shift_a,pert_b+shift_b]

def v_line_locations_from_shift(model:sokm, shift_matrix_obj:ShiftMatrix):
    """
    Calculates the frequencies of shifting eigenvalues prior and post shift for annotation in psd_plots.
    Theoretical post shift frequencies are verified with the existence of an actual resonance frequency in 1e-8 proximity to the predicted shifted resonance frequency. 

    Parameters
    ----------
    model : sokm
        The network model containing the second order Kuramoto model Jacobian and system dynamics.
    shift_matrix_obj : ShiftMatrix
        ShiftMatrix instance containing eigenvalue shifts applied to the model.

    Returns
    -------
    tuple of arrays
        Tuple containing three arrays: (frequencies_prior, frequencies_post_theo, frequencies_post_actual), 
        - frequencies_prior are the resonance frequencies corresponding to the eigenvalues that are shifted, 
        - frequencies_post_theo are the predicted resonance frequencies after applying the shift based on the shift matrix
        - frequencies_post_actual are the actual resonance frequencies of the model after applying the shift
    """

    frequencies_prior= model.predict_resonance_frequencies(indices=shift_matrix_obj.eigenvalue_indices)

    frequencies_post_actual=model.predict_resonance_frequencies(S=shift_matrix_obj.shift_matrix, indices=shift_matrix_obj.eigenvalue_indices) 

    shifted_eigvals=shift_matrix_obj.eigenvalues[shift_matrix_obj.eigenvalue_indices]+shift_matrix_obj.shifts
    frequencies_post_theo=model.predict_resonance_frequencies(shifted_eigvals=shifted_eigvals)
    rel_diffs = np.abs(frequencies_post_theo - frequencies_post_actual) / np.abs(frequencies_post_actual)

    print("Maximum relative difference between predicted post shift frequencies and actual post shift frequencies: ", np.max(rel_diffs)*100, " %.")
    pre_color=np.empty_like(frequencies_prior, dtype=object)
    pre_color.fill("black")
    post_color=np.empty_like(frequencies_post_theo, dtype=object)
    post_color.fill(darkblue)


    return np.concatenate([frequencies_prior, frequencies_post_theo]), np.concatenate([pre_color, post_color])


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
                       fontsize=10):
    shift_matrix_obj = scenario[3][0] if vtn_on else None
    pert_amplitude, pert_freq, pert_node, onset_t= scenario[1]

    transient = len(axes)==3
    t_window_steady= t_window/2 if transient else t_window

    # create grid of subplots with first axis narrower than the others
    save=False
    if axes is None:
        save = True
        fig, axes = plt.subplots(1, 2, figsize=(2.8, 0.6), gridspec_kw={"width_ratios": [3, 3]})
    
    if color.ndim>1:
        scenario_color=blend_colors(color,mode="plt")
    else:
        scenario_color=color

    # calculate trajectories
    traj=generate_or_fetch_scenario_data(t_final, scenario, model, y_0=y_0, steps=steps, meta_scenario_name=scenario_name, save_dir=save_dir, overwrite=overwrite, minus_fixpoint= True,silent=True)
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
    t_traj_index=np.abs(t - t_final + t_window_steady).argmin() 
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
            for i in range(np.shape(psd)[0]):
                axes[1+transient].plot(freqs,psd[i,:],linewidth=linewidth,color=node_colors[i])

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
                axes[1+transient].vlines(freq,0,pert_amplitude, color="black", linewidth=linewidth,alpha=1)
    else:
        S = None
        if vtn_on:
            if type(shift_matrix_obj) is ShiftMatrix:
                S=shift_matrix_obj.shift_matrix
            else:
                S=shift_matrix_obj
        response_vals=model.calculate_response_amplitudes(omega,S=S,k=pert_node)
        axes[1+transient].plot(omega, response_vals[node], color="black",linewidth=linewidth)
        axes[1+transient].set_ylim((np.min(response_vals[node]),10*np.max(response_vals[node])))
    axes[1+transient].set_xlim(min(omega),max(omega))    
    axes[1+transient].set_yscale("log")
    #axes[1+transient].yaxis.tick_right()
    if not only_perturbation:
        axes[1+transient].set_ylim(1e-2,np.max(response_vals)*2)
        #axes[1+transient].set_yticks([round_log(np.min(response_vals[node])),round_log(np.max(response_vals[node])*2)])
        #axes[1+transient].set_yticks([1e-2,1e2])
    else:
        axes[1+transient].set_ylim(1e-2,pert_amplitude*2)
        axes[1+transient].set_yticks([1e-2,pert_amplitude*10])
    if plot_real_psd:
        axes[1+transient].set_ylim((1e-10,10*np.max(response_vals[node])))
    
    # addid perturbation prequency patches
    delta=0.05
    if np.isscalar(pert_freq):
        rect=plt.Rectangle((pert_freq-delta,0), 2*delta, 1e3, color=color,alpha=0.3,edgecolor=[0,0,0,0],linewidth=linewidth)
        axes[1+transient].add_patch(rect)
    else:
        for i, freq in enumerate(pert_freq):
            rect=plt.Rectangle((freq-delta,0), 2*delta, 1e3, color=color[i],alpha=0.3,edgecolor=[0,0,0,0],linewidth=linewidth)
            axes[1+transient].add_patch(rect)

    if only_perturbation:
        if np.isscalar(pert_freq):
            axes[1].vlines(pert_freq,0,pert_amplitude, color="black", linewidth=linewidth,alpha=1)
        else:
            for i, freq in enumerate(pert_freq):
                axes[1].vlines(freq,0,pert_amplitude, color="black", linewidth=linewidth,alpha=1)

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
                             onset_t = 1.
                             ):
    
    # allows for calling with a specific scenario, but also just with a shift matrix object
    if isinstance(scenario, ShiftMatrix): #if no scenario, but only the shiftMatrix object (class) is provided the object construction is visualized  the individual scnearios generated
        shift_matrix_obj=scenario
        compose_shift_matrix_construction_visualization_vertical(shift_matrix_obj, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=overwrite)
        scenario_provided=False
        check_individual_shifts=True
    else:
        shift_matrix_obj=scenario[3][0]
        if isinstance(shift_matrix_obj, ShiftMatrix):
            compose_shift_matrix_construction_visualization_vertical(shift_matrix_obj, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=overwrite)
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


def probe_network(model, shift_matrix_obj=None, pert_node=1,pert_amplitude=0.1, num=100,t_final=800, steps=16000, node=5, save_dir=None, name="large_network", overwrite=True ):
    """
    Creates a plot of the network and two additional panels inderneath side by side with the following content: 
    steady state response to white continuous noise and psd of the steady state response and theoretical psd on top 
    """
    # create subfigure grid
    fig = plt.figure(figsize=(3, 2), layout='constrained')
    axs = fig.subplot_mosaic([["network", "network"],
                          ["trajectory", "psd"]])
    
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
    
    # create the scenario
    print("Creating scenario.")
    all_orig_freqs = model.predict_resonance_frequencies()
    all_orig_freqs = all_orig_freqs[np.invert(np.isnan(all_orig_freqs))]
    pert_freqs=np.linspace(np.min(all_orig_freqs),np.max(all_orig_freqs),num=num)
    pert = (dynamics.sine_perturbation_single_node, (pert_amplitude,pert_freqs,pert_node))

    if vtn_on:
        shift_args = (shift_matrix_obj, model, None)
        shift = (dynamics.jacobian_shift, shift_args)
    else:
        shift=(None,None)    
    scenario=pert+shift

    # fetch trajectory and plot
    print("The simulation bit.")
    traj=generate_or_fetch_scenario_data(t_final, scenario, model, steps=steps, meta_scenario_name="default", overwrite=overwrite, minus_fixpoint=False,silent=False)
    t=traj["t"]
    if vtn_on:
        vals=traj["ys_shifted"][:np.shape(model.jacobian_matrix)[0],:] #fetching only the node trajectories, not the velocity trajectories
    else:
        vals=traj["ys"][:np.shape(model.jacobian_matrix)[0],:]

    #axs["trajectory"].plot(t[-500:],vals[node,-500:])
    pert_vals=np.zeros((np.shape(model.jacobian_matrix)[0],steps+1))
    pert_vals=dynamics.sine_perturbation_single_node(t,pert_vals[:,0],scenario[1])
    print("plotting trajectory")
    axs["trajectory"].plot(t[:300],vals[pert_node,:300])
    axs["trajectory"].plot(t[:300],pert_vals[pert_node,:300],color="red",linewidth=1.)
    
    # calculate and plot psd
    t=t[10000:]
    node_vals=vals[pert_node,10000:]
    
    print("plotting and calculating psd")
    freqs, psd = scipy.signal.welch(node_vals, fs=1/(t[1]-t[0]), axis=0, nperseg=len(t))
    axs["psd"].plot(freqs,psd)
    axs["psd"].set_yscale("log")
    axs["psd"].set_ylim(pert_amplitude/num*1e-2,1.5*np.max(psd))

    # calculate theoretical response amplitudes and plot it on top as a thin black line 
    response_vals=model.calculate_response_amplitudes(freqs,k=pert_node)
    axs["psd"].plot(freqs,response_vals[node,:],color="black",linewidth=1.)
    plt.show()

    # save the figure
    if save_dir== None and shift_matrix_obj==None:
        save_dir= model.current_dir
    elif save_dir== None:
            save_dir = shift_matrix_obj.current_dir
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
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\Example network")
    model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\moritz_a_001")
    #model.compute_jacobian()
    

    # Generate a shift matrix
    shift_matrix_obj = ShiftMatrix(model=model)
    eigenvalue_indices = [ 2,5]
    shifts = np.array([-2.5,-1.5],dtype=float)
    #shifts = np.array([-2.5,-2.5],dtype=float)
    #zero_rows = np.array([5,6,7],dtype=int)
    #zero_rows=np.arange(5, dtype=int)
    zero_rows=np.array([1,2,3,4,5])
    zero_cols = np.array([],dtype=int)
    shift_matrix_obj.construct_from_scratch(eigenvalue_indices, shifts, zero_rows, zero_cols)
    #shift_matrix_obj= ShiftMatrix.load_from_file("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-03-31_10-32-17\\shift_matrix.json")
    #resonance_plot(model,log=True, show_resonance_location=True)
    #pre_and_post_shift_comparison_subplots(model, shift_matrix_obj)
    
    
    print(f"Xs: {shift_matrix_obj.Xs} and Ys: {shift_matrix_obj.Ys}")

    #compose_shift_matrix_construction_visualization(shift_matrix_obj, log=True, absolute=True,fs=8, jac_color=black)
    perturbed_node=5
    t_final=750
    steps=30000
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
    if False:
        print(np.sum(model.power_vector))
        probe_network(model, overwrite=True, pert_amplitude=0.01, num =100)


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
                             labels=True,
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
    