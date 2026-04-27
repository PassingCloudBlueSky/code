from plot_utils import *
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
"""
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'  # Use LaTeX's default serif font
plt.rcParams['text.latex.preamble'] = r'\\usepackage{amsmath}'  # Optional: Add LaTeX packages
"""

import numpy as np
from kuramoto_class import SecondOrderKuramotoModel as sokm 
from shift_matrix import ShiftMatrix
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
                   perturbed_node=6):
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
    for i in range(len(response_vals[:,0])): #loop over all nodes
        plt.plot(omega,response_vals[i,:],alpha=alpha,color=green,linewidth=lw)
        
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
            alpha=0.5

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
                    alpha=1

            # draw vertical line
            plt.axvline(x=current_frequency,linestyle="dashed",color=line_color,alpha=alpha,linewidth=1.)

    
    # Drawing perturbation band if provided
    if pert_band!=None: 
        rect=plt.Rectangle((pert_band[0],0), pert_band[1], 20*np.max(response_vals), color="gold",alpha=0.3)
        ax.add_patch(rect)

    # log or not log that is the question
    if log: 
        #plt.xscale("log")
        plt.yscale("log")
        ax.set_ylim(bottom=0.8*np.min(response_vals))
        plt.ylabel("$A_n$",fontsize=fs)
    else:
        ax.set_ylim(bottom=0)
        plt.ylabel("$A_n$",fontsize=fs)
    
    if x_axis_off:
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    else:
        plt.xlabel("$\omega$",fontsize=fs)

    # aesthetics
    plt.xlim((omega[0],omega[-1]))
    if min_max!=None:
        plt.ylim(min_max)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    for axis in ['top','bottom','left','right']: #thicker axis
        ax.spines[axis].set_linewidth(0.6)
    ax.set_facecolor("white")

    # ascertaining existence of model instance directory
    if model.current_dir == None:
        model.save_parameters()

    save_dir = os.path.join(model.current_dir, "plots")
    save_path=save_figure(fig, save_dir, name=name)
    #plt.show()
    return save_path

    
def plot_network(
    model: sokm,
    shift_matrix_obj: Optional[ShiftMatrix]= None,
    seed=42,
    edge_weight_key="weight",
    vtn_edge_style="dashed",
    name="network.svg", 
    fs=6,
    threshhold=1e-10,
    perturbed_node=None
):
    """
    Plot a network with edge weights represented as thickness and an optional VTN visualization layered on top.

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
    # Create the graph from the connectivity matrix
    G = nx.from_numpy_array(model.connectivity_matrix)

    # Generate positions for the nodes (use seed for reproducibility)
    pos = nx.spring_layout(G, seed=seed)

    # Extract edge weights
    edge_weights = [d.get(edge_weight_key, 1.0) for _, _, d in G.edges(data=True)]

    # Normalize edge weights for visual representation
    max_weight = max(edge_weights) if edge_weights else 1.0
    edge_widths = [2 * (w / max_weight) for w in edge_weights]  # Scale edge thickness
    #edge_opacities = [0.2 + 0.8 * (w / max_weight) for w in edge_weights]  # Scale opacity

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(2, 1.6))

    node_color=darkblue+(1.-darkblue)*0.35
    # Draw the base graph
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=False,
        node_size=100,
        node_color=node_color,
        font_size=fs,
        font_color="black",
        edge_color="black",
        width=edge_widths,
        alpha=1.,  # Base opacity for edges
    )

    # Annotate nodes with their numbers
    nx.draw_networkx_labels(
        G,
        pos,
        labels={node: str(node+1) for node in G.nodes()},  # Label each node with its number
        font_size=fs,
        font_color="black",
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

    # Highlight vtn nodes if provided
    if shift_matrix_obj!=None:
        
        # fetch individual shift matrices and their color coding
        individual_shift_matrices=shift_matrix_obj.cunstruct_individual_shift_matrices(in_eigenspace=False)
        n_individual_shift_matrices=len(individual_shift_matrices)
        shift_colors=create_shift_colors(n_individual_shift_matrices) 
        

        for shift_number in range(n_individual_shift_matrices):

            # collecting all non zero row indices of the individual shift aka the nodes which need active control
            vtn_nodes= np.nonzero(np.any(individual_shift_matrices[shift_number] > threshhold, axis=1))[0]

            print(list(shift_colors[shift_number]))
            # overlying the shift color for the vtn nodes
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=vtn_nodes,
                node_color=list(shift_colors[shift_number]),
                node_size=100,
                alpha=None,
                ax=ax,
            )


            # Fully connect the vtn nodes with dashed edges
            for i, node1 in enumerate(vtn_nodes):
                for node2 in vtn_nodes[i + 1 :]:
                    ax.plot(
                        [pos[node1][0], pos[node2][0]],
                        [pos[node1][1], pos[node2][1]],
                        linestyle=vtn_edge_style,
                        color=shift_colors[shift_number],
                        alpha=0.7,
                        linewidth=np.average(edge_widths)
                    )

    # Save the figure 
    if model.current_dir == None:
        model.save_parameters()
    
    save_dir = os.path.join(model.current_dir, "plots")
    save_path=save_figure(fig, save_dir, name=name)


    return save_path


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


def pre_and_post_shift_comparison_subplots(model:sokm, shift_matrix_obj:ShiftMatrix, log=True, threshhold=1e-10,fs=5,perturbed_node=6):

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

    # fetch unshifted resonance plot data & plot
    print("Visualizing network without VTN...")
    paths.append(resonance_plot(model, pert_band=pert_band, min_max=min_max, name="res.SVG", log=log, extra_scope=0.2, show_resonance_location=True, fs=fs,lw=1,alpha=1.,x_axis_off=True,perturbed_node=perturbed_node))
    # generate shifted resonance data and plot
    print("Visualizing network with VTN...")
    paths.append(resonance_plot(model, shift_matrix_obj=shift_matrix_obj, pert_band=pert_band, min_max=min_max, name= "res_shifted.SVG",log=log, extra_scope=0.2, show_resonance_location=True, fs=fs,lw=1,alpha=1.,perturbed_node=perturbed_node))

    # generate network plot WITH seed
    print("Plotting unshifted response amplitudes...")
    paths.append(plot_network( model, seed=4, name="network.SVG", edge_weight_key="weight",perturbed_node=perturbed_node, threshhold=threshhold))

    # generate network plot with VTN connecting nodes corresponding to non zero rows of the shift matrix
    print("Plotting shifted response amplitudes...")
    paths.append(plot_network( model, shift_matrix_obj=shift_matrix_obj, seed=4, name="network_VTN.SVG", perturbed_node=perturbed_node, threshhold=threshhold ))
    
    return paths


def compose_pre_and_post_shift_comparison(model:sokm, shift_matrix_obj:ShiftMatrix, log=True, vtn_threshhold=1e-10, fontsize=5):

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

    Returns
    -------

    """


    print("Composing network visualizations and response amplitude plots...")
    paths=pre_and_post_shift_comparison_subplots(model, shift_matrix_obj, log=log, threshhold=vtn_threshhold,fs=fontsize)

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
    for i in range(2,4):
        fig_part = sg.fromfile(paths[i])
        network_plot = fig_part.getroot()
        network_plot.moveto((i-2)*260, 0, scale_x=1, scale_y=1)
        if i==2:
            reference=sg.TextElement((i-2)*260+5,5, "a)", size=6)
        else:
            reference=sg.TextElement((i-2)*260+5,5, "d)", size=6)
        fig.append([network_plot,reference])

        
    # saving as SVG and PNG
    save_dir=os.path.join(shift_matrix_obj.current_dir,"plots\\comparison.svg")
    fig.save(save_dir)
    png_path=os.path.join(shift_matrix_obj.current_dir,"plots\\comparison.png")
    svg2png(url=save_dir,write_to=png_path,
            parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)


if __name__ == "__main__":
    
    # Create a model and compute the Jacobian
    model = sokm.from_random_sparse_graph(num_nodes=8, edge_probability=0.3, damping_coefficient=0.01)
    #model.compute_jacobian()
    #model.summary()
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-04-07_11-21-27")

    # Create a ShiftMatrix object
    shift_matrix_obj = ShiftMatrix(model=model)

    # Generate a shift matrix
    eigenvalue_indices = [ 5,6]
    shifts = np.array([-0.6, 0.3])
    zero_rows = np.array([3,4,5,6,7])
    #zero_rows=np.arange(16)
    zero_cols = np.array([],dtype=int)
    shift_matrix_obj.construct_from_scratch(eigenvalue_indices, shifts, zero_rows, zero_cols)

    #resonance_plot(model,log=True, show_resonance_location=True)
    #pre_and_post_shift_comparison_subplots(model, shift_matrix_obj)
    


    
    #shift_matrix_obj= ShiftMatrix.load_from_file("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-03-31_10-32-17\\shift_matrix.json")
    compose_pre_and_post_shift_comparison(model, shift_matrix_obj)

    compose_shift_matrix_construction_visualization(shift_matrix_obj, log=True, absolute=True)