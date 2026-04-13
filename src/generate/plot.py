from typing import ByteString, Optional
from IPython.display import SVG
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib import rc
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
"""
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'  # Use LaTeX's default serif font
plt.rcParams['text.latex.preamble'] = r'\\usepackage{amsmath}'  # Optional: Add LaTeX packages
"""

import matplotlib.image as mpimg
import numpy as np
import numpy.ma as ma
from kuramoto_class import SecondOrderKuramotoModel as sokm 
from shift_matrix import ShiftMatrix
import scipy
import networkx as nx
import os
import svgutils.transform as sg
import svgutils.compose as sc
from cairosvg import svg2png



# Color definitions
blue=np.array([0,170,212,100])/256
darkblue = np.array([0, 68, 170, 100])/256
purple = np.array([205, 135, 222, 100])/256
red = np.array([211, 95, 95, 100])/256
orange = np.array([255, 153, 85, 100])/256
green = np.array([44, 160, 90, 100])/256



def create_cmap_from_white(color, cmap_length=256):
    """
    Create a colormap transitioning from white to the given color.
    """
    cmap_vals = np.ones((cmap_length, 4))
    cmap_vals[:, 0] = np.linspace(1, color[0], cmap_length)
    cmap_vals[:, 1] = np.linspace(1, color[1], cmap_length)
    cmap_vals[:, 2] = np.linspace(1, color[2], cmap_length)
    cmap_vals[:, 3] = np.sqrt(np.linspace(0, 0.9, cmap_length))
    #cmap_vals[:30, 3] = np.linspace(0, 1, 30)
    return ListedColormap(cmap_vals)


def create_shift_colors(num_shifts) -> list:
    """
    Returns a list of colors for the visualization of individual shifts with the length num_shifts.

    Parameters:
    -------
        num_shifts:
            Number of colors to pick, aka number of individual shifts to visualize.
    
    Returns:
    -----
        shift_colors:
            List of RGBA colors of length num_shifts
    
    """
    color_range=[purple, red, orange]
    individual_shift_colors_cmap = LinearSegmentedColormap.from_list("shift_cmap", color_range, N=256)
    shift_colors=[]
    for i in range(num_shifts):
        shift_colors.append(list(individual_shift_colors_cmap (i / (num_shifts - 1))))

    return shift_colors



def save_figure(fig, save_dir: str, name: str) -> str:

        """
        Save the figure 

        
        """

        
        os.makedirs(save_dir, exist_ok=True)

        # creating plot path/name
        #date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
        #save_path=os.path.join(save_dir, date_str+name)
        save_path=os.path.join(save_dir, name)

        fig.savefig(save_path,bbox_inches='tight',format="svg", dpi=300, transparent=True)
        
        return save_path


def generate_frequency_range(model:sokm,extra_scope=0.2):
    """
        Helper function to generate a frequency range for the domain of resonant behavior.
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
                   omega=None, pert_band=None,
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
    Creates a resonance plot for a network with graph Laplacian L and a perturbation at node k. 
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
    seed=None,
    edge_weight_key="weight",
    vtn_edge_style="dashed",
    name="network.svg", 
    fs=6,
    threshhold=1e-10,
    perturbed_node=None
):
    """
    Plot a network with edge weights represented as thickness and an optional VTN visualization.

    Parameters
    ----------
    model : SecondOrderKuramotoModel
        The Kuramoto model containing the connectivity matrix.
    seed : int, optional
        Seed for reproducibility of the layout. Default is None.
    edge_weight_key : str, optional
        The key in the edge attributes that represents the weight. Default is "weight".
    vtn_nodes : list[int], optional
        List of node indices to highlight. These nodes will be fully connected with dashed edges.
        Default is None.
    vtn_node_color : str, optional
        Color for the highlighted nodes. Default is "red".
    vtn_edge_style : str, optional
        Style for the edges connecting the highlighted nodes. Default is "dashed".
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object.
         
    ax : matplotlib.axes.Axes
        The Matplotlib axes object.
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

    # Show the plot
    #plt.show()

    return save_path


def perturbation_band_from_shift_indices(model:sokm, shift_matrix_obj:ShiftMatrix, buffer=0.05):
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
    
    Touple of the left edge value of the perturbation band and the perturbation band's width
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



def plot_shift_comparison(model:sokm, shift_matrix_obj:ShiftMatrix, log=True, threshhold=1e-10,fs=5,perturbed_node=6):


    
    paths=[]
    
    # calculate perturbation band rectangle left and right x values 
    pert_band = perturbation_band_from_shift_indices(model, shift_matrix_obj)
    
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
    paths.append(plot_network( model, seed=4, name="network.SVG", edge_weight_key="weight",perturbed_node=perturbed_node))

    # generate network plot with VTN connecting nodes corresponding to non zero rows of the shift matrix
    print("Plotting shifted response amplitudes...")
    paths.append(plot_network( model, shift_matrix_obj=shift_matrix_obj, seed=4, name="network_VTN.SVG", perturbed_node=perturbed_node ))
    
    return paths


def compose_shift_comparison(model:sokm, shift_matrix_obj:ShiftMatrix, log=True, vtn_treshhold=1e-10, fontsize=5):

    print("Composing network visualizations and response amplitude plots...")
    paths=plot_shift_comparison(model, shift_matrix_obj, log=log, threshhold=vtn_treshhold,fs=fontsize)

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

        

    save_dir=os.path.join(shift_matrix_obj.current_dir,"plots\\comparison.svg")
    fig.save(save_dir)
    png_path=os.path.join(shift_matrix_obj.current_dir,"plots\\comparison.png")
    svg2png(url=save_dir,write_to=png_path,
            parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)



def add_pcolormesh(ax, matrix, color=blue, min_max=None, log=True, cutoff=1e-5, alpha=0.7, absolute=False):
    """
    Helper function to add a pcolormesh to an axis for consistent styling across all higher level functions.
    """

    if log or absolute:
        matrix = np.abs(matrix)

    if min_max==None:
        min_max = (np.min(matrix), np.max(matrix))
    
    # creating log norm if log scaling is desired, with cutoff to avoid issues with log(0)
    if log:
        norm=colors.LogNorm(vmin=min_max[0] if min_max[0]>cutoff else cutoff, vmax=min_max[1], clip=True)
    else:
        norm=Normalize(vmin=min_max[0], vmax=min_max[1], clip=True)

    # adding plot to axis
    print("Shape of matrix:", np.shape(matrix))
    im = ax.pcolormesh(np.flip(matrix,axis=0), cmap=create_cmap_from_white(color), norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
    return im




def visualize_matrix(left_vec=None, 
                     right_vec=None, 
                     matrix=None, 
                     min_max=None,
                     name=None,
                     save_dir=None,
                     color=blue,
                     log=True,
                     absolute=False, 
                     cutoff=1e-5,
                     axs=None,
                     fs=20,
                     label=["left vec", "right vec", "matrix"]):
    """
    Visualize the outer product of two vectors left_vec and right_vec, along with the vectors themselves if left_vec and right_vec are provided.
    If matrix is provded, it visualizes only the matrix provided. The color scale can be adjusted with min_max and cmap.

    Parameters
    ----------
    left_vec : array-like
        First input vector.
    right_vec : array-like
        Second input vector.
    matrix : array-like, optional
        The matrix to visualize. If None, it will be computed from the vectors.
    min_max : tuple, optional
        The minimum and maximum values for the color scale. If None, the values are determined from the data.
    cmap : str, optional
        Colormap for the matrix visualization. Default is "viridis".
    """

   # Input validation
    if (left_vec is None or right_vec is None) and matrix is None:
        raise ValueError("At least left_vec, right_vec, or matrix must be provided.")
    
    # preparatoin of data
    matrix = np.outer(left_vec, right_vec) if matrix is None else matrix
    if log or absolute:
        left_vec=np.abs(left_vec) if left_vec is not None else None
        right_vec=np.abs(right_vec) if right_vec is not None else None
        matrix=np.abs(matrix)
    dim=matrix.shape[0]
    
    # Finding global min and max for consistent color scaling across all three plots if not provided. 
    # For matrices without left_vec and right_vec add_pcolormesh handles the scaling in cases where min_max is not provided.
    if min_max==None and left_vec is not None and right_vec is not None:
        max_val = max(np.max(arr) for arr in [left_vec, right_vec, matrix])
        min_val = min(np.min(arr) for arr in [left_vec, right_vec, matrix])
        min_max = (min_val, max_val)
    

    # generating figure to plot into if no axes for layering onto has been provided
    if axs is None:
        fig, axes = plt.subplots(2,2, width_ratios=(1, dim), height_ratios=( 1,dim), figsize=(1, 1), gridspec_kw=dict(hspace=1/dim, wspace=1/dim))
    else:
        axes=axs

    # add matrix plot
    im=add_pcolormesh(axes[1,1], matrix, color=color, min_max=min_max, log=log, absolute=absolute, cutoff=cutoff)
    axes[1,1].set_xlabel(label[2] if len(label)==3 else label,fontsize=fs)

    # add generating vextor plots if they have been provided as parameters
    if left_vec is not None and right_vec is not None:
        # Reshaping vectors to ensure they are 2D and oriented correctly for the plot
        right_vec=right_vec[None,:] if right_vec.ndim==1 else right_vec
        left_vec=left_vec[:,None] if left_vec.ndim==1 else left_vec
        right_vec=right_vec.T if right_vec.shape[0]>right_vec.shape[1] else right_vec
        left_vec=left_vec.T if left_vec.shape[0]<left_vec.shape[1] else left_vec

        # plotting the vectors
        add_pcolormesh(axes[0,1], right_vec, color=color, min_max=min_max, log=log, absolute=absolute, cutoff=cutoff)
        add_pcolormesh(axes[1,0], left_vec, color=color, min_max=min_max, log=log, absolute=absolute, cutoff=cutoff)

        #labels 
        axes[0,1].set_title(label[1],fontsize=fs)
        axes[1,0].set_ylabel(label[0],fontsize=fs)
    else:
        # empty labels for plots without vectors to keep constistent plot sizes. Eases SVG composing in later stage
        axes[0,1].axis('off')
        axes[0,1].set_title(" ",fontsize=fs)
        axes[1,0].axis('off')
        axes[1,0].set_ylabel(" ",fontsize=fs)
    axes[0,0].axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes((0.95, 0.11, 0.05, 0.77))
    cbar=fig.colorbar(im, cax=cbar_ax, ticks=[min_max[0],min_max[1]])
    cbar.outline.set_linewidth(0.5)
    if min_max!=None:
        #cbar.set_ticks([min_max[0], min_max[1]])  # Set ticks at the min and max values
        cbar.ax.set_yticklabels(["0", "1"],fontsize=fs)   # Label them as 0 and 1
    #cbar.ax.tick_params(labelsize=fs)

    # save figure
    if axs is None:
        if save_dir is None:
            save_dir=os.path.join(os.getcwd(),"unorganized_plots")
        return save_figure(fig, save_dir=save_dir, name=name if name is not None else "pcolormesh.svg")



def construction_visualization(shift_matrix_obj:ShiftMatrix, in_eigenspace=False, log=True, absolute=False, fs=8, cutoff=1e-4):
    """
    Helper function to visualize the construction of the shift matrix by plotting the Jacobian, 
    the individual shift components, and everything layered on top of each other as individual SVGs.

    Parameters
    ----------
    shift_matrix_obj: ShiftMatrix
        The ShiftMatrix object containing the data from which to construct the SVGs.
    in_eigenspace: boolean, optional
        Whether to visualize the construction in the eigenspace of the Jacobian or the physical space.
    log: boolean, optional
        Plotting on log scale the absolute values or not. 
    absolute: boolean, optional
        Plotting absolute values on regular scale.
    cutoff=minimal value to plot. particularly relevant for log-scale plots. 
    fs : int, optional
        Font size for the plot. Default is 20.

    Returns
    ---------
    saving_paths: list of strings
        List containing the paths to all generated SVG figures. Order is Jacobian, individual shift figures, layered plot


    """

    # checking prerequisites
    if shift_matrix_obj.shift_matrix is None:
        raise ValueError("Shift matrix has to be constructed beforehand to be visualized.")

    # Extract data from the ShiftMatrix object
    left_generators, right_generators = shift_matrix_obj.calculate_shift_matrix_generators(in_eigenspace=in_eigenspace)
    individual_shift_matrices= shift_matrix_obj.cunstruct_individual_shift_matrices(in_eigenspace=in_eigenspace)
    if in_eigenspace:
        jacobian= np.diag(shift_matrix_obj.eigenvalues)
    else:
        jacobian = shift_matrix_obj.jacobian
    dim=jacobian.shape[0]
    if shift_matrix_obj.current_dir == None:
        shift_matrix_obj.save_parameters()
    save_dir=os.path.join(shift_matrix_obj.current_dir,"plots")

    # If log scaling is desired, take the absolute value of the data in order to avoid evaluing log of negative values
    if log or absolute:
        left_generators = [np.abs(vec) for vec in left_generators]
        right_generators = [np.abs(vec) for vec in right_generators]
        individual_shift_matrices = [np.abs(mat) for mat in individual_shift_matrices]
        jacobian = np.abs(jacobian)

    # Calculate global min and max for consistent color intensities across all plots
    global_max_val = max(np.max(arr) for arr in [np.concatenate(left_generators), np.concatenate(right_generators), np.concatenate(individual_shift_matrices), jacobian])
    global_min_val = min(np.min(arr) for arr in [np.concatenate(left_generators), np.concatenate(right_generators), np.concatenate(individual_shift_matrices), jacobian])
    min_max = (global_min_val, global_max_val)

    # create plot instances to accumulate the layers of each visualization step for log and physical space
    fig_layered, axes_layered = plt.subplots(2,2, width_ratios=(1, dim), height_ratios=( 1,dim), figsize=(1, 1), gridspec_kw=dict(hspace=1/dim, wspace=1/dim))
    axes_layered[0,1].set_title(" ",fontsize=fs)
    axes_layered[1,0].set_ylabel(" ",fontsize=fs)
    axes_layered[0,1].axis('off')
    axes_layered[1,0].axis('off')
    axes_layered[0,0].axis('off')
    if in_eigenspace:
        axes_layered[1,1].set_xlabel(r"$V(J+\sum_{i}\vec{p}_{i}\vec{q}_i^T)V^{-1}$", fontsize=fs)
    else:
        axes_layered[1,1].set_xlabel(r"$J+\sum_{i}\vec{p}_{i}\vec{q}_i^T$", fontsize=fs)


    # create empty list of saving_paths
    saving_paths=[]

    # Visualize jacobian in eigenspace and physical space
    add_pcolormesh(axes_layered[1,1],jacobian, color=darkblue, log=log, absolute=absolute, cutoff=cutoff, min_max=min_max)
    saving_paths.append(visualize_matrix(matrix=jacobian, label=r"$VJV^{-1}$" if in_eigenspace else r"$ J $", color=darkblue, log=log, absolute=absolute, fs=fs, cutoff=cutoff, name=f"{"eigenspace" if in_eigenspace else "physical"}_jacobian.svg", save_dir=save_dir, min_max=min_max) )
    

    # looping over individual shift matrices to visualize their construction one by one
    n_individual_shift_matrices=len(individual_shift_matrices)
    shift_colors=create_shift_colors(n_individual_shift_matrices) 

    for index in range(n_individual_shift_matrices):
        print(f"Visualizing shift component {index+1} in {'eigen' if in_eigenspace else 'physical'} space...")
        
        # generating labels for subfigures
        p="p"
        q="q"
        if in_eigenspace:
            labels = [rf"$V\vec{{{p}}}_{{{index+1}}}$", rf"$\vec{{{q}}}_{{{index+1}}}^TV^{{{-1}}}$", f"$V\\vec{{{p}}}_{{{index+1}}}\\vec{{{q}}}_{{{index+1}}}^TV^{{{-1}}}$"]
        else:
            labels = [rf"$\vec{{{p}}}_{{{index+1}}}$", rf"$\vec{{{q}}}_{{{index+1}}}^T$", f"$\\vec{{{p}}}_{{{index+1}}}\\vec{{{q}}}_{{{index+1}}}^T$"]
        add_pcolormesh(axes_layered[1,1], individual_shift_matrices[index], color=shift_colors[index], log=log, absolute=absolute, cutoff=cutoff, min_max=min_max)
        saving_paths.append(visualize_matrix(left_vec=left_generators[index], right_vec=right_generators[index],label=labels, color=shift_colors[index], log=log, fs=fs, cutoff=cutoff, name=f"{"eigenspace" if in_eigenspace else "physical"}_shift_component_{index+1}.svg", save_dir=save_dir, min_max=min_max))

    # add row numbers to layered plot
    if in_eigenspace is False:
        
        labels=[]
        for i in range(dim):
            if i%4==0 or i==dim-1:
                    labels.append(f"{dim-i}")
            else:
                labels.append("")

        axes_layered[1,1].set_yticks(np.arange(dim)+0.5)  # Set the positions of the ticks
        axes_layered[1,1].set_yticklabels(labels, fontsize=fs)  # Set the labels and font size


    # save layered figure
    saving_paths.append(save_figure(fig_layered, save_dir=save_dir, name=f"{"eigenspace" if in_eigenspace else "physical"}_layered.svg"))
    plt.close(fig_layered)

    return saving_paths


def construction_publication_ready(shift_matrix_obj:ShiftMatrix, log=True, absolute=True,fs=8):
    
    """
    
    """


    # generate the subfigures based on the properties of shift_matrix_obj
    file_paths_physical=construction_visualization(shift_matrix_obj, in_eigenspace=False, log=log, absolute=absolute, fs=fs, cutoff=1e-4)
    file_paths_eigenspace=construction_visualization(shift_matrix_obj, in_eigenspace=True, log=log, absolute=absolute, fs=fs, cutoff=1e-4)
    
    #create new SVG figure
    fig = sg.SVGFigure("17cm", "6.5cm")
    #fig.append(sc.Grid(10,10)) # visual grid for ease of aligning figures

    # loop to load and add all the generated SVGs to the figure, with appropriate positioning and scaling
    def add_svg_row_to_figure(fig,file_paths, second_row=False):
        """
        Helper function to line up the construction process of a shifted jacobian either in the eigenspace (second_row=False) or physical space (True).
        
        Parameters
        --------
        fig: svgutils Figure
            Object onto which the svg files are appended.
        file_paths: list of strings
            List of the paths to the svg files of the construction process. Starting with the jacobian, the individual shift matrices and ending with the layered figure.
        second_row: boolean
            Indicates current row of cunstruction.
            False: eigenspace
            True: physical space

        Returns
        --------
        save_dir: string
            Path to the finishes SVG file.
        
        """

        # vertical offset of row
        y_0 = 70 if second_row else 10

        # list to store "+"/"=" sign figures such that they can be appended to fi in the end and are layered on top of the other svgs, are not covered by them
        symbols=[]

        #looping over the individual SVGs
        for i,path in enumerate(file_paths):

            # horizontal offset of objects to append
            x_0 = i*70
            if i==0:
                x_0 += 10

            # appending and positioning the svg figure of the current loop
            fig_part = sg.fromfile(path)
            plot = fig_part.getroot()
            plot.moveto(x_0, y_0, scale_x=0.6, scale_y=0.6)
            fig.append([plot])

            # plotting the subfigure reference for the caption, with a bit of extra space for the first and last plot for coherent aesthetics
            if  i==len(file_paths)-1:
                x_0 -= 10
            elif i==0:
                x_0 -=8

            # generating and appending referencing label of the subfigure
            reference_offset=len(file_paths)+1 if second_row else 1
            reference=sg.TextElement(x_0+5,y_0+10, chr(ord('`')+i+reference_offset+4)+")", size=6)

            # generating and appending +/= signs to illustrate the narrative between the figures
            if i<len(file_paths)-2:
                plus=sg.TextElement(x_0+65,y_0+32,"+",size=6)
                symbols.append(plus)
            elif i==len(file_paths)-2:
                equal=sg.TextElement(x_0+65,y_0+32,"=",size=6)
                symbols.append(equal)
            fig.append([reference])
        
        # space label
        space=sg.TextElement(10,y_0+57, "Physical space" if second_row else "Eigenspace", size=6)
        space.rotate(270, 10, y_0+57)
        fig.append([space])

        # append +/= symbols on top of everything else
        fig.append(symbols)
    
    add_svg_row_to_figure(fig, file_paths_eigenspace, second_row=False)
    add_svg_row_to_figure(fig, file_paths_physical, second_row=True)

    # dividing line between the spaces
    line=sc.Line([(5,70),(45+70*(len(file_paths_physical)-1),70)],width=0.8)
    fig.append([line])

    save_dir=os.path.join(shift_matrix_obj.current_dir,"plots\\combined.svg")
    fig.save(save_dir)
    png_path=os.path.join(shift_matrix_obj.current_dir,"plots\\combined.png")
    svg2png(url=save_dir,write_to=png_path,
            parent_height=150,parent_width=70*len(file_paths_eigenspace),output_height=1500,output_width=700*len(file_paths_eigenspace))
    return save_dir




# Example usage
if __name__ == "__main__":
    
    # Create a model and compute the Jacobian
    model = sokm.from_random_sparse_graph(num_nodes=8, edge_probability=0.3, damping_coefficient=0.01)
    #model.compute_jacobian()
    #model.summary()
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-04-07_11-21-27")

    # Create a ShiftMatrix object
    shift_matrix_obj = ShiftMatrix(model=model)

    # Generate a shift matrix
    eigenvalue_indices = [ 2,3,4]
    shifts = np.array([-0.6, -0.5,0.3])
    zero_rows = np.array([3,4])
    #zero_rows=np.arange(16)
    zero_cols = np.array([5,6],dtype=int)
    shift_matrix_obj.construct_from_scratch(eigenvalue_indices, shifts, zero_rows, zero_cols)

    #resonance_plot(model,log=True, show_resonance_location=True)
    #plot_shift_comparison(model, shift_matrix_obj)
    


    
    #shift_matrix_obj= ShiftMatrix.load_from_file("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-03-31_10-32-17\\shift_matrix.json")
    compose_shift_comparison(model, shift_matrix_obj)

    construction_publication_ready(shift_matrix_obj, log=True, absolute=True)
    