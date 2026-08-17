"""
Utility functions for plotting
"""



from typing import Optional
import black
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
plt.rcParams['text.usetex'] = True
"""

plt.rcParams['font.family'] = 'serif'  # Use LaTeX's default serif font
plt.rcParams['text.latex.preamble'] = r'\\usepackage{amsmath}'  # Optional: Add LaTeX packages
"""
#from networkx.drawing.nx_pylab import _FontSize
import numpy as np
from base_model import BaseModel
from kuramoto_class import SecondOrderKuramotoModel as sokm 
from shift_matrix import ShiftMatrix
import networkx as nx
import os
import svgutils.transform as sg
import svgutils.compose as sc
from cairosvg import svg2png
import dynamics
import scipy
import Euler_Maruyama as em



# Color definitions
blue=np.array([0,170,212,100])/256
darkblue = np.array([0, 68, 170, 100])/256
purple = np.array([205, 135, 222, 100])/256
red = np.array([211, 95, 95, 100])/256
orange = np.array([255, 153, 85, 100])/256
green = np.array([44, 160, 90, 100])/256
black = np.array([0, 0, 0, 100])/256

#-----------
# general functions for plotting and saving figures

def create_cmap_from_white(color, cmap_length=256):
    """
    Create a colormap transitioning from white to the given color.

    Parameters:
    -------
        color:
            The target color as an RGBA array (values between 0 and 1).
        cmap_length:
            The number of colors in the colormap. Default is 256.
    Returns:
    -----
        cmap:
            A ListedColormap object transitioning from white to the given color.
    """
    cmap_vals = np.ones((cmap_length, 4))
    cmap_vals[:, 0] = np.linspace(1, color[0], cmap_length)
    cmap_vals[:, 1] = np.linspace(1, color[1], cmap_length)
    cmap_vals[:, 2] = np.linspace(1, color[2], cmap_length)
    cmap_vals[:, 3] = np.sqrt(np.linspace(0, 0.9, cmap_length))
    #cmap_vals[:30, 3] = np.linspace(0, 1, 30)
    return ListedColormap(cmap_vals)

def blend_colors(colors, mode="average",default_alpha=0.8):
    a=colors[:,3]
    blended_color=np.zeros((4,))
    
    if mode=="average":
        n=len(colors[0,:])
        for i in range(3):
            blended_color[i]=np.sum(colors[:,i]*a[:])/n
        blended_color[3]=np.sum(a[:])/n
    elif mode=="screen":
        blended_color[3]=1-np.prod(1-a[:])
        for i in range(3):
            blended_color[i]=1-np.prod(1-colors[:,i]*a[:])
    elif mode=="plt":
        blended_color=colors[0,:]

        def blend_two_colors(RGBold, RGBnew):
            #helperfunction to blend two colors
            alpha = 0.5
            return RGBold * (1 - alpha) + RGBnew * alpha
        
        # iteratively layering the colors on top of each other
        for i in range(np.shape(colors)[0]-1):
            blended_color=blend_two_colors(blended_color, colors[i+1,:])


    return blended_color

def create_shift_colors(num_shifts, flipped=False) -> list:
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
    
    color_range=[orange, red, purple]
    individual_shift_colors_cmap = LinearSegmentedColormap.from_list("shift_cmap", color_range, N=256)
    shift_colors=[]
    #picking colors from the cmap
    if num_shifts==1:
        shift_colors.append(list(individual_shift_colors_cmap(0.)))
    else:
        for i in range(num_shifts):
            shift_colors.append(list(individual_shift_colors_cmap (i / (num_shifts - 1))))

    return shift_colors


def node_colors_from_perturbation_distance(perturbation_source: int, model:BaseModel, cmap=plt.cm.viridis, weight="weight"):
    """
    Generate colors for nodes based on their distance from a perturbation source.

    Parameters:
    -------
        perturbation_source:
            The node index from which to calculate distances.
        model:
            The model containing the jacobian matrix.
        cmap:
            The colormap to use for generating colors. Default is plt.cm.viridis.

    Returns:
    -------
        node_colors:
            A list of RGBA colors for each node based on their distance from the perturbation source.
    """

    # convert network to binary
    G = nx.from_numpy_array((model.jacobian_matrix!=0).astype(int))
    distances = nx.shortest_path(G,source=perturbation_source,weight=weight)
    #print("Distances from perturbation source node:", distances)
    distances= [sum(distances[i]) for i in range(len(distances))]
    
    # pick colors from cmap based on distance to perturbation source node
    node_colors = [cmap(distances[i]/max(distances)) for i in range(len(distances))]
    #node_colors=np.insert(node_colors, perturbation_source, [cmap(0)], axis=0) # set color of perturbation source node to the color corresponding to distance 0
    # return list of node colors
    return node_colors

def round_log(a, decimals=0):
    """
    Function for space efficient log scale plotting. Probably already exists in some library but was quicker and more fun to quickly code. 
    Returns the input float rounded to the leading non zero order. 
    Decimals is the number of decimals to which it is rounded in the leading order.
    E.g. decimals=2, a= 0.07356 -> 0.0736
    decimals=0, a=0.07356 -> 0.07
    """
    for i, val in enumerate(a):
        n=1
        while np.abs(val/(10**(n)))>=10:
            n+=1
        while np.abs(val/(10**(n)))<=1:
            n-=1
        a[i]=np.round(val/10**n, decimals=decimals)*10**n
    return a

def save_figure(fig, save_dir: str, name: str) -> str:

        """
        Save the figure to the provided directory under the given name.

        Parameters:
        -------
            fig:
                The matplotlib figure to save.
            save_dir:
                The directory where the figure should be saved.
            name:
                The name of the file to save the figure as.
        
        Returns:
        -------
            save_path:
                The full path to the saved figure.
        
        """

        
        os.makedirs(save_dir, exist_ok=True)
        save_path=os.path.join(save_dir, name)

        fig.savefig(save_path,bbox_inches='tight',format="svg", dpi=300, transparent=True)
        
        return save_path


#-----------
# for shift matrix construction visualization


def add_pcolormesh(ax, matrix, color=blue, min_max=None, log=True, cutoff=1e-5, absolute=False, cmap=None):
    """
    Helper function to add a pcolormesh to an axis for consistent styling across all higher level functions.

    Parameters:
    -------
        ax:
            The matplotlib axis to which the pcolormesh should be added.
        matrix:
            The matrix to visualize as a pcolormesh on top of the provided matrix.
        color:
            The color to use for the pcolormesh. Default is blue.
        min_max:
            Tuple of (min, max) values for the color scale. If None, it will be determined from the data. Default is None.
        log:
            Whether to use logarithmic scaling for the color intensity. Default is True.
        cutoff:
            The minimum value to plot when using logarithmic scaling. Values below this will be set to the cutoff value to avoid issues with log(0). Default is 1e-5.
        abosulute:
            Whether to take the abosulte value for plotting without plotting log. Default is False.
    
    Returns:
    -------
        im:
            The matplotlib image object created by pcolormesh, which can be used for further customization or adding a colorbar.
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
    if cmap is None:
        cmap=create_cmap_from_white(color)
    im = ax.pcolormesh(np.flip(matrix,axis=0), cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
    return im


def visualize_matrix(left_vec=None, 
                     right_vec=None, 
                     matrix=None, 
                     min_max=None,
                     name: Optional[str]=None,
                     save_dir: Optional[str]=None,
                     color=blue,
                     log=True,
                     absolute=False, 
                     cutoff=1e-5,
                     axs=None,
                     fs=20,
                     label=["left vec", "right vec", "matrix","a)","space"],
                     size=(1.5, 1.5)):
    """
    Visualize the outer product of two vectors left_vec and right_vec, along with the vectors themselves if left_vec and right_vec are provided.
    If matrix is provded, it visualizes only the matrix provided. The color scale can be adjusted with min_max and color. Save_dir and name can be provided to save the figure.
    If log is true a logarithmic color scale is used, if absolute is true the absolute values are plotted on a regular scale. Cutoff can be provided to avoid issues with log(0) 
    when log is true. If axs is provided, the plots are added to the provided axes for layering onto an already existing figure, otherwise a new figure is generated. 
    fs is the fontsize and label is the label for the left vector, right vector and matrix plot respectively. If only one label is provided, it is used for the matrix plot.

    Parameters
    ----------
    left_vec : array-like
        Left construction vector.
    right_vec : array-like
        Right construction vector.
    matrix : array-like, optional
        The matrix to visualize. If None, it will be computed as the outer product of left_vec and right_vec.
    min_max : tuple, optional
        The minimum and maximum values for the color scale. If None, the values are determined from the data.
    name: str
        The name to save the figure under if save_dir is provided. Default is None, which will use "pcolormesh.svg".
    save_dir: str
        The directory to save the figure in. Default is None, which will use a directory named "unorganized_plots" in the current working directory.
    color: array-like
        The RGBA color to use for the pcolormesh. Default is blue.
    log: bool
        Whether to use logarithmic scaling for the color intensity. Default is True.
    absolute: bool
        Whether to plot the absolute values (relevant for non-log plots). Default is False.
    cutoff: float
        The cutoff value for logarithmic scaling. Default is 1e-5.
    axs: matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is generated.
    fs: int
        The fontsize for the labels. Default is 20.
    label: list of str
        The labels for the left vector, right vector and matrix plot respectively. If only one label is provided, it is used for the matrix plot.
    
    returns
    -------
    If axs is None, the path to the saved figure. Otherwise, None.
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
        fig, axes = plt.subplots(2,2, width_ratios=(1, dim), height_ratios=( 1,dim), figsize=size, gridspec_kw=dict(hspace=1/dim, wspace=1/dim))
    else:
        axes=axs

    # add matrix plot
    im=add_pcolormesh(axes[1,1], matrix, color=color, min_max=min_max, log=log, absolute=absolute, cutoff=cutoff)
    axes[1,1].set_xlabel(label[2],fontsize=fs)

    # add generating vector plots if they have been provided as parameters
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
        if len(label)>4:
            print(f"plotting {label[4]} label")
            axes[1,0].set_ylabel(label[4],fontsize=fs)
        else:
            axes[1,0].set_ylabel("",fontsize=fs)
    axes[0,0].axis('off')
    axes[1,1].annotate(rf"$\mathrm{{{label[3]}}}$", (-1.7, 8.6), fontsize=fs, annotation_clip=False)

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


def shift_matrix_construction_visualization_subplots(shift_matrix_obj:ShiftMatrix, space="physical", log=True, absolute=False, fs=10, cutoff=1e-4, jac_color=darkblue, node_reference = False, size=(1.5, 1.5), flipped=False):
    """
    Helper function to create the subplots for the composed figure visualizing the construction of the shift matrix by plotting the Jacobian, 
    the individual shift components, and everything layered on top of each other as individual SVGs. 

    Parameters
    ----------
    shift_matrix_obj: ShiftMatrix
        The ShiftMatrix object containing the data from which to construct the SVGs.
    in_eigenspace: boolean, optional
        Whether to visualize the construction in the eigenspace of the Jacobian or the physical space.
    log: boolean, optional
        Plotting on log scale the absolute values. Default if True. 
    absolute: boolean, optional
        Plotting absolute values on regular scale.
    cutoff: float, optional
        The cutoff value for logarithmic scaling. Default is 1e-5.
    fs : int, optional
        Font size for the plot. Default is 8.

    Returns
    ---------
    saving_paths: list of strings
        List containing the paths to all generated SVG figures. Order is Jacobian, individual shift figures, layered plot


    """

    # checking prerequisites
    if shift_matrix_obj.shift_matrix is None:
        raise ValueError("Shift matrix has to be constructed beforehand to be visualized.")
    

    # Extract data from the ShiftMatrix object
    permutation, left_generators, right_generators = shift_matrix_obj.calculate_shift_matrix_generators(space=space)
    matrix_permutation, individual_shift_matrices= shift_matrix_obj.construct_individual_shift_matrices(space=space)
    #print(f"inside the visualization the individual shift matrices are {individual_shift_matrices}")
    n_individual_shift_matrices=len(individual_shift_matrices)
    if space is "eigen":
        jacobian= np.diag(shift_matrix_obj.eigenvalues)
    elif space is "physical":
        jacobian = shift_matrix_obj.jacobian
    elif space is "permutation":
        jacobian= np.diag(shift_matrix_obj.eigenvalues)
        jacobian = jacobian[matrix_permutation]
        
    dim=jacobian.shape[0]

    shift_colors=create_shift_colors(n_individual_shift_matrices) 

    # flipping the ordering or the rows and columns in order to hav ethe same visual ordering to the resonance frequencies
    if flipped:
        left_generators=np.flip(left_generators)
        right_generators=np.flip(right_generators)
        individual_shift_matrices=np.flip(individual_shift_matrices)
        jacobian=np.flip(jacobian)
        shift_colors=np.flip(shift_colors,axis=0)


    if shift_matrix_obj.current_dir == None:
        shift_matrix_obj.save_parameters()
    save_dir=shift_matrix_obj.current_dir

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
    fig_layered, axes_layered = plt.subplots(2,2, width_ratios=(1, dim), height_ratios=( 1,dim), figsize=size, gridspec_kw=dict(hspace=1/dim, wspace=1/dim))
    axes_layered[0,1].set_title(" ",fontsize=fs)
    axes_layered[1,0].set_ylabel(" ",fontsize=fs)
    axes_layered[0,1].axis('off')
    axes_layered[1,0].axis('off')
    axes_layered[0,0].axis('off')
    if space is "eigen":
        axes_layered[1,1].set_xlabel(r"$VJV^{-1}$"+"\n"+r"+$V\sum_{i}\vec{p}_{i}\vec{q}_i^T)V^{-1}$", fontsize=fs)
        #axes_layered[1,1].set_xlabel(r"$V(J+\sum_{i}\vec{p}_{i}\vec{q}_i^T)V^{-1}$", fontsize=fs)
        panel_label=rf"{chr(97+2*n_individual_shift_matrices+3)}"
    elif space is "physical":
        axes_layered[1,1].set_xlabel(r"$J+\sum_{i}\vec{p}_{i}\vec{q}_i^T$", fontsize=fs)
        panel_label=rf"{chr(97+n_individual_shift_matrices+1)}"
    elif space is "permutation":
        #axes_layered[1,1].set_xlabel(r"$\pi VJ(\pi V)^{-1}$"+"\n"+r"$+\pi V\sum_{i}\vec{p}_{i}\vec{q}_i^T(\pi V)^{-1}$", fontsize=fs)
        axes_layered[1,1].set_xlabel(r"$\widetilde VJ\widetilde V^{-1}$"+"\n"+r"+$\widetilde V\sum_{i}\vec{p}_{i}\vec{q}_i^T)\widetilde V^{-1}$", fontsize=fs)
        #axes_layered[1,1].set_xlabel(r"$\widetilde V(J\widetilde +\sum_{i}\vec{p}_{i}\vec{q}_i^T))\widetilde V^{-1}$", fontsize=fs)
        panel_label=rf"{chr(97+3*n_individual_shift_matrices+5)}"
    axes_layered[1,1].annotate(rf"\textbf{{{panel_label}}}", (-1.5, 8.5), fontsize=fs, annotation_clip=False)
    #visualize_matrix(matrix=jacobian, axs=axes_layered, color=jac_color, log=log, absolute=absolute, fs=fs, cutoff=cutoff, min_max=min_max, size=size)

    # create empty list of saving_paths
    saving_paths=[]

    # Visualize jacobian in eigenspace and physical space
    add_pcolormesh(axes_layered[1,1],jacobian, color=jac_color, log=log, absolute=absolute, cutoff=cutoff, min_max=min_max)
    if space is "eigen":
        panel_label=rf"{chr(97+n_individual_shift_matrices+2)}"
        labels=[None,None,r"$VJV^{-1}$",rf"\textbf{{{panel_label}}}",r"eigenspace \n of $J$"]
    elif space is "physical":
        panel_label=rf"{chr(97)}"
        labels=[None,None,r"$J$",rf"\textbf{{{panel_label}}}",r"physical space"]
    elif space is "permutation":
        panel_label=rf"{chr(97+2*n_individual_shift_matrices+4)}"
        #labels=[None,None,r"$\pi VJ(\pi V)^{-1}$",rf"\textbf{{{panel_label}}}", r"\permuted \n eigenspace \n of $J$"]
        labels=[None,None,r"$\widetilde VJ\widetilde V^{-1}$",rf"\textbf{{{panel_label}}}", r"\permuted \n eigenspace \n of $J$"]
    saving_paths.append(visualize_matrix(matrix=jacobian, label=labels, color=jac_color, log=log, absolute=absolute, fs=fs, cutoff=cutoff, name=f"{space}_jacobian.svg", save_dir=save_dir, min_max=min_max, size=size))

    # looping over individual shift matrices to visualize their construction one by one

    for index in range(n_individual_shift_matrices):
        print(f"Visualizing shift component {index+1} in {space}-space...")
        

        # generating labels for subfigures
        p="p"
        q="q"
        if space is "eigen":
            panel_label=rf"{chr(100+n_individual_shift_matrices+index)}"
            labels = [rf"$V\vec{{{p}}}_{{{index+1}}}$",
                       rf"$\vec{{{q}}}_{{{index+1}}}^TV^{{{-1}}}$",
                         f"$V\\vec{{{p}}}_{{{index+1}}}\\vec{{{q}}}_{{{index+1}}}^TV^{{{-1}}}$",
                         rf"\textbf{{{panel_label}}}"]
        elif space is "physical":
            panel_label=rf"{chr(98+index)}"
            labels = [rf"$\vec{{{p}}}_{{{index+1}}}$", 
                      rf"$\vec{{{q}}}_{{{index+1}}}^T$", 
                      f"$\\vec{{{p}}}_{{{index+1}}}\\vec{{{q}}}_{{{index+1}}}^T$",
                      rf"\textbf{{{panel_label}}}"]
        if space is "permutation":
            panel_label=rf"{chr(102+2*n_individual_shift_matrices+index)}"
            if False:
                labels = [rf"$\pi V\vec{{{p}}}_{{{index+1}}}$",
                       rf"$\vec{{{q}}}_{{{index+1}}}^TV^{{{-1}}}\pi^{{{-1}}}$",
                         f"$\\pi V\\vec{{{p}}}_{{{index+1}}}\\vec{{{q}}}_{{{index+1}}}^TV^{{{-1}}}\\pi^{{{-1}}}$",
                         rf"\textbf{{{panel_label}}}"]
                labels = [rf"$\pi V\vec{{{p}}}_{{{index+1}}}$",
                                   rf"$\vec{{{q}}}_{{{index+1}}}^T(\pi V)^{{{-1}}}$",
                                     f"$\\pi V\\vec{{{p}}}_{{{index+1}}}\\vec{{{q}}}_{{{index+1}}}^T(\pi V)^{{{-1}}}$",
                                     rf"\textbf{{{panel_label}}}"]
            labels = [rf"$\widetilde V\vec{{{p}}}_{{{index+1}}}$",
                                   rf"$\vec{{{q}}}_{{{index+1}}}^T\widetilde V^{{{-1}}}$",
                                     f"$\\widetilde V\\vec{{{p}}}_{{{index+1}}}\\vec{{{q}}}_{{{index+1}}}^T\\widetilde V^{{{-1}}}$",
                                     rf"\textbf{{{panel_label}}}"]
        add_pcolormesh(axes_layered[1,1], individual_shift_matrices[index], color=shift_colors[index], log=log, absolute=absolute, cutoff=cutoff, min_max=min_max)
        saving_paths.append(visualize_matrix(left_vec=left_generators[index], right_vec=right_generators[index],label=labels, color=shift_colors[index], log=log, fs=fs, cutoff=cutoff, name=f"{space}_shift_component_{index+1}.svg", save_dir=save_dir, min_max=min_max,size=size))

    # add row numbers to layered plot
    if node_reference and space is "physical" and dim <=10:
        
        labels=[]
        for i in range(dim):
            if i%4==0 or i==dim-1:
                    labels.append(f"{dim-i}")
            else:
                labels.append("")

        axes_layered[1,1].set_yticks(np.arange(dim)+0.5)  # Set the positions of the ticks
        axes_layered[1,1].set_yticklabels(labels, fontsize=fs)  # Set the labels and font size


    # save layered figure
    saving_paths.append(save_figure(fig_layered, save_dir=save_dir, name=f"{space}_layered.svg"))
    plt.close(fig_layered)

    return saving_paths


def compose_shift_matrix_construction_visualization_horizontal(shift_matrix_obj:ShiftMatrix, log=True, absolute=True,fs=8, jac_color=darkblue, overwrite=False, type="thumbnail",flipped=False, equation_mode= True, show_permutation=True):
    """
    Compose a comprehensive visualization of the shift matrix construction process.
    This function generates a side-by-side comparison of the shift matrix construction
    in both eigenspace and physical space, combining multiple subplot visualizations
    into a single SVG figure with labeled stages and operators. The combined visualization is saved in both SVG
    and PNG formats in the plots subdirectory.
    ----------
    shift_matrix_obj : ShiftMatrix
        The shift matrix object containing construction data and output directory information.
    log : bool, optional
        If True, apply logarithmic scaling to the visualizations. Default is True.
    absolute : bool, optional
        If True, display absolute values and log False, display absolute values on linear scale.
    fs : int, optional
        Font size for the plots. Default is 8.
    jac_color : str, optional
        Color for the Jacobian matrix visualization. Default is darkblue.
    overwrite : bool, optional
        If True, overwrite existing files. Default is False.
    type : str, optional
        Type of the visualization. Default is "thumbnail".
    -------
    save_dir : str
        Path to the saved combined SVG file (plots\\combined.svg).
    """

    # checking overwrite
    save_dir=os.path.join(shift_matrix_obj.current_dir,f"combined_horizontal_{type}{"_permutation" if show_permutation else ""}.svg")
    if overwrite is False and os.path.isfile(save_dir):
        return save_dir

    #create new SVG figure
    if type=="thumbnail":
        combined_fig = sg.SVGFigure("3.5in","1.8in") #("17cm", "6.5cm")
        size=(0.85,0.85)
        scale=80
    elif type=="shift":
        combined_fig = sg.SVGFigure("4.3in","2.2in")
        size=(1.25,1.25)
        scale=100
    elif type=="new":
        combined_fig = sg.SVGFigure("3.5in","2in")
        size=(1.1,1.1)
        scale=90

    #grid for alignment
        #combined_fig.append(sc.Grid(20,20)) # visual grid for ease of aligning figures

    # generate the subfigures based on the properties of shift_matrix_obj
    file_paths_physical=shift_matrix_construction_visualization_subplots(shift_matrix_obj, space="physical", log=log, absolute=absolute, fs=fs, cutoff=1e-4, jac_color=jac_color, size=size,flipped=flipped)
    file_paths_eigenspace=shift_matrix_construction_visualization_subplots(shift_matrix_obj, space="eigen", log=log, absolute=absolute, fs=fs, cutoff=1e-4, jac_color=jac_color, size=size,flipped=flipped)
    if show_permutation:
        file_paths_permutation=shift_matrix_construction_visualization_subplots(shift_matrix_obj, space="permutation", log=log, absolute=absolute, fs=fs, cutoff=1e-4, jac_color=jac_color, size=size,flipped=flipped)
    

    # helper function to loop over, load and add all the generated SVGs to the figure, with appropriate positioning and scaling
    def add_svg_row_to_figure(fig,file_paths, row_count=0, scale=80,equation_mode=True):
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
        y_0 = scale*1.05*row_count

        # list to store "+"/"=" sign figures such that they can be appended to fi in the end and are layered on top of the other svgs, are not covered by them
        symbols=[]

        #looping over the individual SVGs
        x_0=0
        for i,path in enumerate(file_paths):

            # horizontal offset of objects to append
            if i>0:
                if equation_mode:
                    x_0 += scale + 10 
                else:
                    x_0 += scale

            # appending and positioning the svg figure of the current loop
            fig_part = sg.fromfile(path)
            plot = fig_part.getroot()
            plot.moveto(x_0, y_0, scale_x=1., scale_y=1.)
            fig.append([plot])

            # plotting the subfigure reference for the caption, with a bit of extra space for the first and last plot for coherent aesthetics
            if  i==len(file_paths)-1:
                x_0 -= 10
            elif i==0:
                x_0 -=8
            
            if equation_mode:
                # generating and appending +/= signs to illustrate the narrative between the figures
                if i<len(file_paths)-2:
                    plus=sg.TextElement(x_0+scale+5,y_0+scale/2+10,"+",size=10)
                    symbols.append(plus)
                elif i==len(file_paths)-2:
                    equal=sg.TextElement(x_0+scale+5,y_0+scale/2+10,"=",size=10)
                    symbols.append(equal)
                fig.append(symbols)
        if row_count > 0:
            line=sc.Line([(0,scale*1.1*row_count),(scale*len(file_paths_physical),scale*1.1*row_count)],width=0.8)
            combined_fig.append([line])

                
    add_svg_row_to_figure(combined_fig, file_paths_eigenspace, row_count=1,scale=scale, equation_mode=equation_mode)
    add_svg_row_to_figure(combined_fig, file_paths_physical, row_count=0,scale=scale, equation_mode=equation_mode)
    if show_permutation:
        add_svg_row_to_figure(combined_fig, file_paths_permutation, row_count=2,scale=scale, equation_mode=equation_mode)
    # dividing line between the spaces
    if equation_mode:
        h_scale = scale+ 10
    

    row_count=2    
    if show_permutation:
        row_count=3
    combined_fig.save(save_dir)
    png_path=os.path.join(shift_matrix_obj.current_dir,f"combined_horizontal_{type}_{"permutation" if show_permutation else ""}.png")
    svg2png(url=save_dir,write_to=png_path,
            parent_height=scale*1.1*row_count,parent_width=scale*(len(file_paths_eigenspace)),output_height=scale*1.1*row_count,output_width=1.05*scale*(len(file_paths_eigenspace)))
    return save_dir


def compose_shift_matrix_construction_visualization_vertical(shift_matrix_obj:ShiftMatrix, log=True, absolute=True,fs=10, jac_color=darkblue, overwrite=False, equation_mode=False):
    """
    Compose a comprehensive visualization of the shift matrix construction process.
    This function generates a side-by-side comparison of the shift matrix construction
    in both eigenspace and physical space, combining multiple subplot visualizations
    into a single SVG figure with labeled stages and operators. The combined visualization is saved in both SVG
    and PNG formats in the plots subdirectory.
    ----------
    shift_matrix_obj : ShiftMatrix
        The shift matrix object containing construction data and output directory information.
    log : bool, optional
        If True, apply logarithmic scaling to the visualizations. Default is True.
    absolute : bool, optional
        If True, display absolute values and log False, display absolute values on linear scale.
    fs : int, optional
        Font size for the plots. Default is 8.
    jac_color : str, optional
        Color for the Jacobian matrix visualization. Default is darkblue.
    overwrite : bool, optional
        If True, overwrite existing files. Default is False.
    -------
    save_dir : str
        Path to the saved combined SVG file (plots\\combined.svg).
    """

    # checking overwrite
    save_dir=os.path.join(shift_matrix_obj.current_dir,"combined_vertical.svg")
    if overwrite is False and os.path.isfile(save_dir):
        return save_dir

    # generate the subfigures based on the properties of shift_matrix_obj
    file_paths_physical=shift_matrix_construction_visualization_subplots(shift_matrix_obj, in_eigenspace=False, log=log, absolute=absolute, fs=fs, cutoff=1e-4, jac_color=jac_color)
    file_paths_eigenspace=shift_matrix_construction_visualization_subplots(shift_matrix_obj, in_eigenspace=True, log=log, absolute=absolute, fs=fs, cutoff=1e-4, jac_color=jac_color)
    
    #create new SVG figure
    fig = sg.SVGFigure("3.25in","6.5in") #("17cm", "6.5cm")
    #fig.append(sc.Grid(20,20)) # visual grid for ease of aligning figures

    v_spacing= 140 if equation_mode else 120

    # helper function to loop over, load and add all the generated SVGs to the figure, with appropriate positioning and scaling
    def add_svg_column_to_figure(fig,file_paths, second_column=False, equation_mode=False):
        """
        Helper function to line up the construction process of a shifted jacobian either in the eigenspace (second_row=False) or physical space (True).
        
        Parameters
        --------
        fig: svgutils Figure
            Object onto which the svg files are appended.
        file_paths: list of strings
            List of the paths to the svg files of the construction process. Starting with the jacobian, the individual shift matrices and ending with the layered figure.
        second_colum: boolean
            Indicates current column of cunstruction.
            False: eigenspace
            True: physical space

        Returns
        --------
        save_dir: string
            Path to the finishes SVG file.
        
        """

        # vertical offset of row
        x_0 = 140 if second_column else 0

        # list to store "+"/"=" sign figures such that they can be appended to fi in the end and are layered on top of the other svgs, are not covered by them
        symbols=[]

        v_spacing= 140 if equation_mode else 120
        #looping over the individual SVGs
        for i,path in enumerate(file_paths):

            

            # vertical offset of objects to append
            y_0 = i*v_spacing

            # appending and positioning the svg figure of the current loop
            fig_part = sg.fromfile(path)
            plot = fig_part.getroot()

            if  i==len(file_paths)-1:
                y_0 -= 20
            plot.moveto(x_0 + 15 if i==0 or i==len(file_paths)-1 else x_0, y_0, scale_x=1., scale_y=1.)
            fig.append([plot])

            if equation_mode:
                # generating and appending +/= signs to illustrate the narrative between the figures
                if i<len(file_paths)-2:
                    plus=sg.TextElement(67+x_0,y_0+133,"+",size=10)
                    symbols.append(plus)
                elif i==len(file_paths)-2:
                    equal=sg.TextElement(67+x_0,y_0+133,"=",size=10)
                    symbols.append(equal)
        
        # append +/= symbols on top of everything else
        fig.append(symbols)
    
    add_svg_column_to_figure(fig, file_paths_eigenspace, second_column=True)
    add_svg_column_to_figure(fig, file_paths_physical, second_column=False)

    # dividing line between the spaces
    line=sc.Line([(140,0),(140,v_spacing*(len(file_paths_physical))-20)],width=0.8)
    fig.append([line])

    fig.save(save_dir)
    png_path=os.path.join(shift_matrix_obj.current_dir,"combined_vertical.png")
    svg2png(url=save_dir,write_to=png_path,
            parent_height=140*len(file_paths_eigenspace)-20,parent_width=280,output_height=140*len(file_paths_eigenspace)-20,output_width=280)
    return save_dir

#-----------
# for stability analysis

def plot_angles_between_eigenvectors(matrix, ax, name="angles_between_eigenvectors", title="matrix", save_dir=None,fs=8):

    # fetch eigenvectors
    eigenValues, eigenVectors = np.linalg.eig(matrix)

    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    # calcualte angle matrix between eigenvectors
    angle_matrix = np.empty((eigenVectors.shape[1], eigenVectors.shape[1]))
    for i in range(eigenVectors.shape[1]):
        for j in range(eigenVectors.shape[1]):
            angle_matrix[i, j] = np.arccos(np.clip(np.dot(eigenVectors[:, i], eigenVectors[:, j]) /
                                      (np.linalg.norm(eigenVectors[:, i]) * np.linalg.norm(eigenVectors[:, j])), -1.0, 1.0))
            
    # visualize angle matrix as pcolormesh

    add_pcolormesh(ax, angle_matrix, cmap=plt.cm.Reds_r, min_max=(0, np.pi/2), log=False, cutoff=1e-5, absolute=True)
    ax.set_title(title, fontsize=fs)


def angle_comparison(model: sokm, shift_matrix_obj: ShiftMatrix, name="angle_comparison", save_dir=None):

    # initialize plot
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # angle plot prior to shift
    plot_angles_between_eigenvectors(model.jacobian_matrix, ax=axes[0], title="w/o shift")

    # angle plot after shift
    plot_angles_between_eigenvectors(model.jacobian_matrix+shift_matrix_obj.shift_matrix, ax=axes[1], title="with shift")

    # add colorbar
    cbar_ax = fig.add_axes((0.95, 0.11, 0.05, 0.77))
    cbar=fig.colorbar(axes[1].collections[0], cax=cbar_ax)
    cbar.outline.set_linewidth(0.5)

    # save figure
    if save_dir is None:
        save_dir = shift_matrix_obj.current_dir
    svg_path=save_figure(fig, save_dir=save_dir, name=name+".svg")
    png_path=os.path.join(save_dir,name+".png")
    svg2png(url=svg_path,write_to=png_path,
            parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)


#-----------
# for dynamics

def generate_or_fetch_scenario_data(t_final: float, args, model: BaseModel, y_0=None, steps=8000, meta_scenario_name="default",
                    save_dir: Optional[str]=None, overwrite=False, minus_fixpoint=False,silent=False, euler_maruyama=False):

    """
    Checks if the trajectory data of the scenario (w and w/o) VTN already exists. If so it is loaded. Else generated and saved. 
    """

    # unpacking arguments
    instance = (model.model_ode, model.model_ode_args)
    pert = args[0:2]
    print("per: ", pert)
    shift = args[2:4]

    # checking if data already exists
    if save_dir == None:
        if shift[1] is None:
            save_dir=model.current_dir
        else:
            save_dir=shift[1][0].current_dir


    filepath = os.path.join(save_dir,meta_scenario_name)+"_tfinal"+str(t_final)+"_tonset"+str(int(pert[1][3]))
    if euler_maruyama:
        filepath += "_euler_maruyama"
    filepath += ".npz"

    

    if overwrite is False and os.path.isfile(filepath):
        if not silent:
            print(f"Loading trajectory data from {filepath}")
        traj = np.load(filepath, allow_pickle=True)
        return traj
    
    if not silent:
        print(f"Simulating trajectory data for {filepath}, since none was found or overwrite is True.")

    
    dim = np.shape(model.jacobian_matrix)[0]

    # set y_0 to fixpoint if not provided
    if y_0 is None:
        if model.fixed_point is None:
            model.compute_fixed_point()
        fixpoint = model.fixed_point
        y_0 = np.zeros(model.ode_dimension)
        y_0[:dim] = fixpoint

    # numerically integrating shifted and unshifted trajectories
    if euler_maruyama:
        t,ys, meta = em.use_euler_maruyama(model, [0,t_final], dt= t_final/steps, y0=y_0, pert=pert, shift_matrix_object=None, seed=42)
        ys=ys.T
    else:
        t, ys = dynamics.integrate_f(t_final, y_0, instance+pert+(None,None), t_0=0, steps=steps)
    if shift[1] is not None:
        t, ys_shifted = dynamics.integrate_f(t_final, y_0, instance+pert+shift, t_0=0, steps=steps)


    
    #plt.show()
    if False:
        # fix point deviation
        ys[:len(y_0)] -= y_0[:, None]
        ys_shifted[:len(y_0)] -= y_0[:, None]

    if shift[1] is None:
        np.savez(filepath, t=t, ys=ys-y_0[:,None] if minus_fixpoint else ys)
    else:
        np.savez(filepath, t=t, ys=ys-y_0[:,None] if minus_fixpoint else ys, ys_shifted=ys_shifted- y_0[:,None] if minus_fixpoint else ys_shifted)
    
    return np.load(filepath, allow_pickle=True)


def load_time_series_data(file_path):
    """
    Converts time series data from a text file into two NumPy arrays:
    - Time in seconds starting from 0
    - Corresponding measured values
    
    Args:
        file_path (str): Path to the text file containing the time series data.
    
    Returns:
        tuple: Two NumPy arrays (time_in_seconds, measured_values)
    """
    # Load the data from the file
    data = np.loadtxt(file_path, delimiter=',', dtype=str)
    
    # Extract time and values
    time_strings = data[:, 1]  # Second column (time)
    values = data[:, 2].astype(float)  # Third column (measured values)
    
    # Convert time strings to seconds
    time_in_seconds = np.array([
        int(h) * 3600 + int(m) * 60 + int(s)
        for h, m, s in (t.split(':') for t in time_strings)
    ])
    
    # Normalize time to start from 0
    time_in_seconds -= time_in_seconds[0]
    
    return time_in_seconds, values


def time_series_and_psd(file_path,name="trajectory"):

    t, y=load_time_series_data(file_path)
    #t,y,outlier_indices=clean_time_series(t, y, method='modified_zscore', threshold=3.5)
    #print(f"Removed outlier indices are: {outlier_indices}")
    t=t[y>30]
    y=y[y>30]

    freqs, psd = scipy.signal.welch(y, fs=1/(t[1]-t[0]), axis=0, nperseg=len(t))

    fig, axes = plt.subplots(1, 2, figsize=(3., 0.7))

    axes[0].plot(t[0:1000],y[0:1000],linewidth=0.5)
    axes[1].plot(freqs,psd,linewidth=0.5)
    #axes[1].set_xlim((0.,0.025))
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    save_dir="C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\50Hz\\"
    svg_path=save_figure(fig, save_dir=save_dir, name=name+".svg")
    png_path=os.path.join(save_dir,name+".png")
    svg2png(url=svg_path,write_to=png_path,
                parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)


def compute_psd_from_traj(t, vals, steady_state_t=500):
    """
    Computes the power spectral density from a trajectory using scipy.signal.welch.
    """
    if np.max(t)< steady_state_t-100:
        print(f"WARNING: The function making this print is agnostic of your chosen alpha value, however if it is 0.01 as it should be for realistic power grids," \
                "and the transient go with e**(-alpha *t) equilibrium with transients only one percent of the original size do not occur before time ~t>460. "\
                    "the maximum time is {np.max(t)}, so the sample size for steady-state power spectral density computation is either non existent or critically small."\
                        "Hence the first entry considered for psd compuation is set to {np.max(t)-100}.")
        steady_state_t= np.max(t)-100
    steady_idx=np.argmin(np.abs(t-steady_state_t))
    t=t[steady_idx:]
    vals=vals[:,steady_idx:]
    freqs, psd = scipy.signal.welch(vals, fs=1/(t[1]-t[0]), axis=-1, nperseg=len(t))
    return freqs, psd

# Example usage
if __name__ == "__main__":
    
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Moritz_vals")
    #model=sokm.from_adjacency_matrix("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\Example network\\adjacency.dat", a=0.01,p_c=-1.,p_p=3.,edge_weight=16)
    #model.compute_jacobian()
    #model.save_parameters()
    # Generate a shift matrix
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
    #compose_shift_matrix_construction_visualization_vertical(shift_matrix_obj, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=True)
    compose_shift_matrix_construction_visualization_horizontal(shift_matrix_obj, log=True, absolute=True,fs=12, jac_color=darkblue,overwrite=True,type="new",flipped=True, show_permutation=True)
    compose_shift_matrix_construction_visualization_horizontal(shift_matrix_obj, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=True,type="shift",flipped=True, show_permutation=False)
    
    time_series_and_psd("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\50Hz\\201105_Frequenz.txt")


    
    #shift_matrix_obj= ShiftMatrix.load_from_file("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-03-31_10-32-17\\shift_matrix.json")

    #compose_shift_matrix_construction_visualization(shift_matrix_obj, log=True, absolute=True)
    