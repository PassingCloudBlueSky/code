from typing import Optional
from IPython.display import SVG
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib import rc
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
import numpy as np
import numpy.ma as ma
from kuramoto_class import SecondOrderKuramotoModel as sokm 
from shift_matrix import ShiftMatrix
import scipy
import networkx as nx
import os
import svgutils.transform as sg
import svgutils.compose as sc
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


def save_figure(fig, save_dir: str, name: str) -> str:

        """
        Save the figure 

        
        """

        
        os.makedirs(save_dir, exist_ok=True)

        # creating plot path/name
        #date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
        #save_path=os.path.join(save_dir, date_str+name)
        save_path=os.path.join(save_dir, name)

        fig.savefig(save_path,bbox_inches='tight',format="svg", dpi=300)
        
        return save_path




def resonance_plot(model: sokm,a=0.01,log=False, extra_scope=0.1, show_resonance_location=False, fs=15,lw=1,alpha=0.8):
    """
    creates a resonance plot for a network with graph Laplacian L and a perturbation at node k. 
    """
    #calculate eigenvalues and eigenvectors
    eigvals, eigvec= scipy.linalg.eigh(model.jacobian_matrix)
    
    
    # generate frequency array adapted to the frequency range with resonant behaviour
    resonance_frequencies=model.predict_resonance_frequencies()
    resonance_frequencies=resonance_frequencies[np.invert(np.isnan(resonance_frequencies))] #throwing out the np.nan caused by 0 eigenvalue
    lower_bound=np.min(resonance_frequencies)
    upper_bound=np.max(resonance_frequencies)
    puffer=extra_scope*(upper_bound-lower_bound)
    lower_bound-=puffer
    upper_bound+=puffer
    omega=np.linspace(lower_bound,upper_bound,num=1000) #frequency values over which we plot


    response_vals=model.calculate_response_amplitudes(omega)
    

    #loop over all nodes, calculate their frequency dependent response and plot them into one graph
    fig,ax=plt.subplots(1, 1,figsize=(14,4.6))
    for i in range(len(eigvals)): #loop over all nodes
        plt.plot(omega,response_vals[i,:],alpha=alpha,color="royalblue",linewidth=lw)
        
    # dashed lines for analytic resonance location
    if show_resonance_location:
        for i in range(len(resonance_frequencies)):
            plt.axvline(x=resonance_frequencies[i],linestyle="dashed",color="grey",alpha=0.5,linewidth=1.)

    """
    # Drawing perturbation band if provided
    if np.any(box!=0): 
        eigvals=np.flip(eigvals)
        box=pred_resonance(eigvals[box],a)
        #print(box)
        print("box width: ",box[1]-box[0])
        ax.add_patch(Rectangle((0.99*box[1],0), 0.8*(box[0]-box[1]), 10*np.max(np.vstack((response_vals,response_vals_hat))),color="gold",alpha=0.5,label="perturbation band"))
    """

    # log or not log that is the question
    if log: 
        plt.xscale("log")
        plt.yscale("log")
        ax.set_ylim(bottom=0.8*np.min(response_vals))
        plt.ylabel("$\log(A_n)$",fontsize=fs)
    else:
        ax.set_ylim(bottom=0)
        plt.ylabel("$A_n$",fontsize=fs)

    # aesthetics
    plt.xlim((omega[0],omega[-1]))
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    for axis in ['top','bottom','left','right']: #thicker axis
        ax.spines[axis].set_linewidth(1.2)
    plt.xlabel("$\omega$",fontsize=fs)

    # ascertaining existence of model instance directory
    if model.current_dir == None:
        model.save_parameters()


    save_dir = os.path.join(model.current_dir, "plots")
    save_figure(fig, save_dir, name="resonance_plot.svg")
    plt.show()


    
def plot_network(
    model: sokm,
    vtn_nodes: Optional[np.ndarray]= None,
    seed=None,
    edge_weight_key="weight",
    vtn_node_color="orange",
    vtn_edge_style="dashed",
    save_path=None,
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
    edge_widths = [8 * (w / max_weight) for w in edge_weights]  # Scale edge thickness
    #edge_opacities = [0.2 + 0.8 * (w / max_weight) for w in edge_weights]  # Scale opacity

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw the base graph
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=False,
        node_size=800,
        node_color="skyblue",
        font_size=10,
        font_color="black",
        edge_color="gray",
        width=edge_widths,
        alpha=1.,  # Base opacity for edges
    )

    # Highlight specific nodes if provided
    if np.any(vtn_nodes!=None):
        # Draw highlighted nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=vtn_nodes,
            node_color=vtn_node_color,
            node_size=800,
            ax=ax,
        )

        # Fully connect the highlighted nodes with dashed edges
        for i, node1 in enumerate(vtn_nodes):
            for node2 in vtn_nodes[i + 1 :]:
                ax.plot(
                    [pos[node1][0], pos[node2][0]],
                    [pos[node1][1], pos[node2][1]],
                    linestyle=vtn_edge_style,
                    color=vtn_node_color,
                    alpha=0.7,
                    linewidth=np.average(edge_widths)
                )

    # Save the figure 
    if model.current_dir == None:
        model.save_parameters()
    
    save_dir = os.path.join(model.current_dir, "plots")
    save_figure(fig, save_dir, name="network.svg")

    # Show the plot
    plt.show()

    return fig, ax



def add_pcolormesh(ax, matrix, color=blue, min_max=None, log=True, cutoff=1e-5, alpha=0.7, absolute=False):
    """
    Helper function to add a pcolormesh to an axis for consistent styling across higher level functions.
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
    

    # Plotting
    if axs is None:
        fig, axes = plt.subplots(2,2, width_ratios=(1, dim), height_ratios=( 1,dim), figsize=((dim+2)/10, (dim+2)/10), gridspec_kw=dict(hspace=1/dim, wspace=1/dim))
    else:
        axes=axs

    
    im=add_pcolormesh(axes[1,1], matrix, color=color, min_max=min_max, log=log, absolute=absolute, cutoff=cutoff)
    axes[1,1].set_xlabel(label[2] if len(label)==3 else label,fontsize=fs)
    if left_vec is not None and right_vec is not None:
        # Reshaping vectors to ensure they are 2D and oriented correctly for the plot
        right_vec=right_vec[None,:] if right_vec.ndim==1 else right_vec
        left_vec=left_vec[:,None] if left_vec.ndim==1 else left_vec
        right_vec=right_vec.T if right_vec.shape[0]>right_vec.shape[1] else right_vec
        left_vec=left_vec.T if left_vec.shape[0]<left_vec.shape[1] else left_vec

        add_pcolormesh(axes[0,1], right_vec, color=color, min_max=min_max, log=log, absolute=absolute, cutoff=cutoff)
        add_pcolormesh(axes[1,0], left_vec, color=color, min_max=min_max, log=log, absolute=absolute, cutoff=cutoff)

        axes[0,1].set_title(label[1],fontsize=fs)
        axes[1,0].set_ylabel(label[0],fontsize=fs)
    else:
        axes[0,1].axis('off')
        axes[0,1].set_title(" ",fontsize=fs)
        axes[1,0].axis('off')
        axes[1,0].set_ylabel(" ",fontsize=fs)
    axes[0,0].axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes((0.95, 0.11, 0.05, 0.77))
    cbar=fig.colorbar(im, cax=cbar_ax)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=fs-1)

    # save figure
    if axs is None:
        if save_dir is None:
            save_dir=os.path.join(os.getcwd(),"unorganized_plots")
        return save_figure(fig, save_dir=save_dir, name=name if name is not None else "pcolormesh.svg")



def construction_visualization(shift_matrix_obj:ShiftMatrix, in_eigenspace=False, log=True, absolute=False, fs=8, cutoff=1e-4):
    """
    Visualize the construction of the shift matrix by plotting the Jacobian, the individual shift components, and the final shift matrix.

    Parameters
    ----------
    shift_matrix : ShiftMatrix
        The ShiftMatrix object containing the shift matrix and related data.
    fs : int, optional
        Font size for the plot. Default is 20.
    """

    if shift_matrix_obj.shift_matrix is None:
        raise ValueError("Shift matrix has to be constructed beforehand to be visualized.")

    # Extract data from the ShiftMatrix object
    left_generators, right_generators = shift_matrix_obj.calculate_shift_matrix_generators(in_eigenspace=in_eigenspace)
    individual_shift_matrices= [np.outer(p_i, q_i) for p_i, q_i in zip(left_generators, right_generators)]
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
    fig_layered, axes_layered = plt.subplots(2,2, width_ratios=(1, dim), height_ratios=( 1,dim), figsize=((dim+2)/10, (dim+2)/10), gridspec_kw=dict(hspace=1/dim, wspace=1/dim))
    axes_layered[0,1].set_title(" ",fontsize=fs)
    axes_layered[1,0].set_ylabel(" ",fontsize=fs)
    axes_layered[0,1].axis('off')
    axes_layered[1,0].axis('off')
    axes_layered[0,0].axis('off')

    # create empty list of saving_paths
    saving_paths=[]

    # Visualize jacobian in eigenspace and physical space
    add_pcolormesh(axes_layered[1,1],jacobian, color=darkblue, log=log, absolute=absolute, cutoff=cutoff, min_max=min_max)
    saving_paths.append(visualize_matrix(matrix=jacobian, label="J", color=darkblue, log=log, absolute=absolute, fs=fs, cutoff=cutoff, name=f"{"eigenspace" if in_eigenspace else "physical"}_jacobian.svg", save_dir=save_dir, min_max=min_max) )
    

    # Visualize individual shift matrices in eigen and physical space
    color_range=[purple, red, orange]
    individual_shift_colors_cmap = LinearSegmentedColormap.from_list("shift_cmap", color_range, N=256)

    n_individual_shift_matrices=len(individual_shift_matrices)
    for index in range(n_individual_shift_matrices):
        print(f"Visualizing shift component {index+1} in {'eigen' if in_eigenspace else 'physical'} space...")
        color=individual_shift_colors_cmap (index / (n_individual_shift_matrices - 1))

        p="p"
        q="q"
        labels = [rf"$\vec{{{p}}}_{{{index+1}}}$", rf"$\vec{{{q}}}_{{{index+1}}}^T$", f"$S_{index+1}\\coloneq\\vec{{{p}}}_{{{index+1}}}\\vec{{{q}}}_{{{index+1}}}^T$"]

        add_pcolormesh(axes_layered[1,1], individual_shift_matrices[index], color=color, log=log, absolute=absolute, cutoff=cutoff, min_max=min_max)
        saving_paths.append(visualize_matrix(left_vec=left_generators[index], right_vec=right_generators[index],label=labels, color=color, log=log, fs=fs, cutoff=cutoff, name=f"{"eigenspace" if in_eigenspace else "physical"}_shift_component_{index+1}.svg", save_dir=save_dir, min_max=min_max))
    
    axes_layered[1,1].set_xlabel(r"$J+\sum_{i}S_i$", fontsize=fs)
    # save layered figure
    saving_paths.append(save_figure(fig_layered, save_dir=save_dir, name=f"{"eigenspace" if in_eigenspace else "physical"}_layered.svg"))
    plt.close(fig_layered)
    return saving_paths


def construction_publication_ready(shift_matrix_obj:ShiftMatrix, log=True, absolute=True,fs=8):
    

    file_paths_physical=construction_visualization(shift_matrix_obj, in_eigenspace=False, log=log, absolute=absolute, fs=fs, cutoff=1e-4)
    file_paths_eigenspace=construction_visualization(shift_matrix_obj, in_eigenspace=True, log=log, absolute=absolute, fs=fs, cutoff=1e-4)
    
    #create new SVG figure
    fig = sg.SVGFigure("17cm", "6.5cm")
    #fig.append(sc.Grid(10,10))

    # loop to load and add all the generated SVGs to the figure, with appropriate positioning and scaling
    symbols=[]
    def add_svg_to_figure(fig,file_paths, second_row=False):
        
        y_0 = 64 if second_row else 10
        for i,path in enumerate(file_paths):
            fig_part = sg.fromfile(path)
            plot = fig_part.getroot()
            x_0 = i*70
            if i==0:
                x_0 += 10
            
            plot.moveto(x_0, y_0, scale_x=0.6, scale_y=0.6)
            fig.append([plot])

            # plotting the subfigure reference for the caption, with a bit of extra space for the first and last plot for coherent aesthetics
            if  i==len(file_paths)-1:
                x_0 -= 10
            elif i==0:
                x_0 -=8
            reference_offset=len(file_paths)+1 if second_row else 1
            reference=sg.TextElement(x_0+10,y_0+15, chr(ord('`')+i+reference_offset)+")", size=6)
            if i<len(file_paths)-2:
                plus=sg.TextElement(x_0+65,y_0+32,"+",size=6)
                symbols.append(plus)
                #fig.append([plus])
            elif i==len(file_paths)-2:
                equal=sg.TextElement(x_0+65,y_0+32,"=",size=6)
                symbols.append(equal)
                #fig.append([equal])
            fig.append([reference])
        
        space=sg.TextElement(10,y_0+50, "Physical space" if second_row else "Eigenspace", size=6)
        space.rotate(270, 10, y_0+50)
        fig.append([space])
        fig.append(symbols)
    
    add_svg_to_figure(fig, file_paths_eigenspace, second_row=False)
    add_svg_to_figure(fig, file_paths_physical, second_row=True)

    line=sc.Line([(5,66),(45+70*(len(file_paths_physical)-1),66)],width=0.8)
    fig.append([line])

    save_dir=os.path.join(shift_matrix_obj.current_dir,"plots\\combined.svg")
    fig.save(save_dir)
    return save_dir

"""
    # load matpotlib-generated figures
    # loop over paths and add them to the figure
    fig1 = sg.fromfile('outer_product_visualization.svg')
    fig2 = sg.fromfile('output.svg')

    # get the plot objects
    plot1 = fig1.getroot()
    plot2 = fig2.getroot()
    plot2.moveto(20, 0, scale_x=0.5)

    # add text labels
    txt1 = sg.TextElement(25,20, "A", size=12, weight="bold")
    txt2 = sg.TextElement(305,20, "B", size=12, weight="bold")

    # append plots and labels to figure
    fig.append([plot1, plot2])
    fig.append([txt1, txt2])

    # save generated SVG files
    fig.save("fig_final.svg")

    Figure("16cm", "6.5cm", 
            Panel(
                SVG("outer_product_visualization.svg"),
                Text("A", 25, 20, size=12, weight='bold')
                ).move(30,0),
            Panel(
                SVG("output.svg").scale(0.5),
                Text("B", 25, 20, size=12, weight='bold')
                ).move(20, 0),
                Grid(20,20)
            ).save("fig_final_compose.svg")
            """

# Example usage
if __name__ == "__main__":
    
    # Create a model and compute the Jacobian
    model = sokm.from_random_sparse_graph(num_nodes=8, edge_probability=0.3, damping_coefficient=0.01)
    model.compute_jacobian()
    model.summary()
    vtn_nodes=np.array([1,3,5,6])
    #plot_network(model,vtn_nodes=vtn_nodes)
    #resonance_plot(model,log=True, show_resonance_location=True)

    
    # Create a ShiftMatrix object
    shift_matrix_obj = ShiftMatrix(model=model)

    # Generate a shift matrix
    eigenvalue_indices = [ 4,5]
    shifts = np.array([-0.5, 0.3])
    zero_rows = np.array([4,5,6,7])
    zero_cols = np.array([0])
    shift_matrix_obj.construct_from_scratch(eigenvalue_indices, shifts, zero_rows, zero_cols)
    

    #shift_matrix_obj= ShiftMatrix.load_from_file("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-03-24_11-40-42\\shift_matrix.json")
    # Visualize the shift matrix
   # visualize_shift_matrixes_save(shift_matrix_obj, log=True)
    #visualize_shift_matrixes_eigenspace_save(shift_matrix_obj, log=True)
    #isualize_matrix(matrix=shift_matrix_obj.shift_matrix)
    #visualize_matrix(left_vec=a[None,:], right_vec=b[:, None], matrix=None, min_max=None, color=green)
    #construction_visualization(shift_matrix_obj, in_eigenspace=False, absolute=True, log=False, fs=20, cutoff=1e-5)
    #construction_visualization(shift_matrix_obj, in_eigenspace=True, absolute=True, log=False, fs=20, cutoff=1e-5)
    construction_publication_ready(shift_matrix_obj, log=True, absolute=True)
    