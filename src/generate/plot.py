from typing import Optional
from IPython.display import SVG
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
import numpy.ma as ma
from kuramoto_class import SecondOrderKuramotoModel as sokm 
from shift_matrix import ShiftMatrix
import scipy
import networkx as nx
import os
import datetime

# Color definitions
blue=np.array([0,170,212,100])
darkblue = np.array([0, 68, 170, 100])
purple = np.array([205, 135, 222, 100])
red = np.array([211, 95, 95, 100])
orange = np.array([255, 153, 85, 100])
green = np.array([44, 160, 90, 100])


def create_cmap_from_white(color, cmap_length=256):
    """
    Create a colormap transitioning from white to the given color.
    """
    cmap_vals = np.ones((cmap_length, 4))
    cmap_vals[:, 0] = np.linspace(1, color[0] / 256, cmap_length)
    cmap_vals[:, 1] = np.linspace(1, color[1] / 256, cmap_length)
    cmap_vals[:, 2] = np.linspace(1, color[2] / 256, cmap_length)
    cmap_vals[:30, 3] = np.linspace(0, 1, 30)
    return ListedColormap(cmap_vals)


def save_figure(fig, save_dir: str, name: str) -> str:

        """
        Save the figure 

        
        """

        
        save_dir = os.path.join(save_dir, "plots")
        os.makedirs(save_dir, exist_ok=True)

        # creating plot path/name
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
        save_path=os.path.join(save_dir, date_str+name)

        fig.savefig(save_path,bbox_inches='tight',format="svg", dpi=300)
        
        return save_dir



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

    save_figure(fig, model.current_dir, name="resonance_plot.svg")
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
    save_figure(fig, model.current_dir, name="network.svg")

    # Show the plot
    plt.show()

    return fig, ax



def add_pcolormesh(ax, matrix,cmap,norm=None):
    """
    Helper function to add a pcolormesh to an axis with consistent styling.
    """
    im = ax.pcolormesh(matrix, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.2)
    return im




def visualize_matrix(left_vec=None, right_vec=None, 
                     matrix=None, 
                     min_max=None, 
                     cmap=create_cmap_from_white(blue), 
                     absolute_values=True, 
                     log=True, 
                     cutoff=1e-5):
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
    matrix = np.outer(left_vec, right_vec) if matrix is None else matrix
    dim=matrix.shape[0]
    if log is True: absolute_values=True 
    if absolute_values:
        if left_vec is not None and right_vec is not None:
            right_vec=np.abs(right_vec)
            left_vec=np.abs(left_vec)
        matrix=np.abs(matrix)

    # Initialization
    fig, axes = plt.subplots(2,2, width_ratios=(1, dim), height_ratios=( 1,dim), figsize=(dim+2, dim+2), gridspec_kw=dict(hspace=1/dim, wspace=1/dim))
    

    # colormap setup
    if min_max==None:
        if left_vec is not None and right_vec is not None:
            max_val = max(np.max(arr) for arr in [left_vec, right_vec, matrix])
            min_val = min(np.min(arr) for arr in [left_vec, right_vec, matrix])
        else:
            max_val = np.max(matrix)
            min_val = np.min(matrix)
    else:
        min_val, max_val = min_max
    
    # creating log norm if log scaling is desired, with cutoff to avoid issues with log(0)
    if log:
        log_norm=colors.LogNorm(vmin=min_val if min_val>cutoff else cutoff, vmax=max_val, clip=True)

    # Plotting
    im=add_pcolormesh(axes[1,1], matrix, cmap=cmap, norm=log_norm if log else None)
    if left_vec is not None and right_vec is not None:
        add_pcolormesh(axes[0,1], right_vec[None, :], cmap=cmap, norm=log_norm if log else None)
        add_pcolormesh(axes[1,0], left_vec[:, None], cmap=cmap, norm=log_norm if log else None)
    else:
        axes[0,1].axis('off')
        axes[1,0].axis('off')
    axes[0,0].axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes((0.95, 0.11, 0.05, 0.77))
    cbar=fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=25)

    # save figure
    plt.savefig("pcolormesh.svg", format="svg", dpi=300, bbox_inches='tight')
    plt.show()
    return fig, axes




def visualize_multiple(sm: ShiftMatrix, in_eigenspace=True, log=True, fs=20):
    """
    Visualize the shift matrix and its components in the physical basis.

    Parameters
    ----------
    sm : ShiftMatrix
        The ShiftMatrix object containing the shift matrix and related data.
    log : bool, optional
        Whether to use logarithmic scaling for the visualization. Default is True.
    fs : int, optional
        Font size for the plot. Default is 20.
    """
    if sm.shift_matrix is None:
        raise ValueError("Shift matrix has not been constructed.")

    # Extract data from the ShiftMatrix object
    S = sm.shift_matrix
    Xs = sm.Xs
    Ys = sm.Ys
    etas = sm.etas
    nus = sm.nus
    jacobian = sm.jacobian
    eigvecs = sm.eigenvectors

    return 


def visualize_shift_matrixes_save(shift_matrix_obj: ShiftMatrix, log=True, fs=20):
    """
    Visualize the shift matrix and its components in the physical basis.

    Parameters
    ----------
    shift_matrix_obj : ShiftMatrix
        The ShiftMatrix object containing the shift matrix and related data.
    log : bool, optional
        Whether to use logarithmic scaling for the visualization. Default is True.
    fs : int, optional
        Font size for the plot. Default is 20.
    """
    if shift_matrix_obj.shift_matrix is None:
        raise ValueError("Shift matrix has not been constructed.")

    # Extract data from the ShiftMatrix object
    S = shift_matrix_obj.shift_matrix
    Xs = shift_matrix_obj.Xs
    Ys = shift_matrix_obj.Ys
    etas = shift_matrix_obj.etas
    nus = shift_matrix_obj.nus
    jacobian = shift_matrix_obj.jacobian
    eigvecs = shift_matrix_obj.eigenvectors

    # Initialize plots
    colors = [darkblue, purple, red, orange]
    titles = ["$\mathcal{L}$", "$S_1$", "$S_2$", "$S_3$", "$\mathcal{L}+S$"]
    cbars = ["$\mathrm{log}_{10}(\mathcal{L})$", "$\mathrm{log}_{10}(S_1)$", "$\mathrm{log}_{10}(S_2)$", "$\mathrm{log}_{10}(S_3)$"]
    file_names = ["L", "S1", "S2", "S3", "LnS"]

    # Visualize the Jacobian (jacobian)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.2))
    m_jacobian = ma.masked_array(jacobian, mask=(np.abs(jacobian) < 1e-10))
    visualize_matrix(m_jacobian, ax, color=colors[0], log=log)
    fig.colorbar(ax.pcolor(m_jacobian), ax=ax, label=cbars[0])
    plt.title(titles[0], fontsize=fs)
    plt.savefig(file_names[0], bbox_inches='tight')

    # Visualize each component of the shift matrix
    for index in range(len(Xs)):
        a = Xs[index]
        p = eigvecs[:, a] @ etas[index]
        b = Ys[index]
        q = eigvecs[:, b] @ nus[index]
        S_current = np.outer(p, q)

        fig, ax = plt.subplots(1, 1, figsize=(7, 5.2))
        m_S_current = ma.masked_array(S_current, mask=(np.abs(S_current) < 1e-10))
        visualize_matrix(m_S_current, ax, color=colors[index + 1], log=log)
        fig.colorbar(ax.pcolor(m_S_current), ax=ax, label=cbars[index + 1])
        plt.title(titles[index + 1], fontsize=fs)
        plt.savefig(file_names[index + 1], bbox_inches='tight')

    # Visualize the full shift matrix
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.2))
    m_S = ma.masked_array(S, mask=(np.abs(S) < 1e-10))
    visualize_matrix(m_S, ax, color=colors[-1], log=log)
    fig.colorbar(ax.pcolor(m_S), ax=ax, label="$\mathrm{log}_{10}(\mathcal{L}+S)$")
    plt.title("$\mathcal{L}+S$", fontsize=fs)
    plt.savefig(file_names[-1], bbox_inches='tight')


def visualize_shift_matrixes_eigenspace_save(shift_matrix_obj: ShiftMatrix, log=True, fs=20):
    """
    Visualize the shift matrix and its components in the eigenbasis.

    Parameters
    ----------
    shift_matrix_obj : ShiftMatrix
        The ShiftMatrix object containing the shift matrix and related data.
    log : bool, optional
        Whether to use logarithmic scaling for the visualization. Default is True.
    fs : int, optional
        Font size for the plot. Default is 20.
    """
    if shift_matrix_obj.shift_matrix is None:
        raise ValueError("Shift matrix has not been constructed.")

    # Extract data from the ShiftMatrix object
    S = shift_matrix_obj.shift_matrix
    Xs = shift_matrix_obj.Xs
    Ys = shift_matrix_obj.Ys
    etas = shift_matrix_obj.etas
    nus = shift_matrix_obj.nus
    jacobian = shift_matrix_obj.jacobian
    eigvals, eigvecs = shift_matrix_obj.eigenvalues, shift_matrix_obj.eigenvectors

    # Transform shift matrix to eigenbasis
    S_eigenbasis = eigvecs.T @ S @ eigvecs

    # Visualize the shift matrix in the eigenbasis
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.2))
    m_S_eigenbasis = ma.masked_array(S_eigenbasis, mask=(np.abs(S_eigenbasis) < 1e-10))
    visualize_matrix(m_S_eigenbasis, ax, color=darkblue, log=log)
    fig.colorbar(ax.pcolor(m_S_eigenbasis), ax=ax, label="$\mathrm{log}_{10}(VSV^{-1})$")
    plt.title("$VSV^{-1}$", fontsize=fs)
    plt.savefig("S_eigenbasis", bbox_inches='tight')


# Example usage
if __name__ == "__main__":
    # Create a model and compute the Jacobian
    model = sokm.from_random_sparse_graph(num_nodes=10, edge_probability=0.3, damping_coefficient=0.01)
    model.compute_jacobian()
    model.summary()
    vtn_nodes=np.array([1,3,5,6])
    #plot_network(model,vtn_nodes=vtn_nodes)
    #resonance_plot(model,log=True, show_resonance_location=True)

    # Create a ShiftMatrix object
    shift_matrix_obj = ShiftMatrix(model=model)

    # Generate a shift matrix
    eigenvalue_indices = [1, 2]
    shifts = np.array([-0.5, 0.3])
    zero_rows = np.array([8, 9])
    zero_cols = np.array([8, 9])
    shift_matrix_obj.construct_from_scratch(eigenvalue_indices, shifts, zero_rows, zero_cols)

    # Visualize the shift matrix
   # visualize_shift_matrixes_save(shift_matrix_obj, log=True)
    #visualize_shift_matrixes_eigenspace_save(shift_matrix_obj, log=True)
    visualize_matrix(matrix=shift_matrix_obj.shift_matrix)
    #a=np.random.rand(10)**3
    #b=np.random.rand(10)
    #visualize_matrix(left_vec=a, right_vec=b, matrix=None, min_max=None, cmap=create_cmap_from_white(green), absolute_values=True)
    