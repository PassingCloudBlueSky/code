"""
Utility functions for plotting
"""



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
import dynamics
import scipy.linalg as la



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
    #picking colors from the cmap
    for i in range(num_shifts):
        shift_colors.append(list(individual_shift_colors_cmap (i / (num_shifts - 1))))

    return shift_colors


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


def add_pcolormesh(ax, matrix, color=blue, min_max=None, log=True, cutoff=1e-5, absolute=False):
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
                     name: Optional[str]=None,
                     save_dir: Optional[str]=None,
                     color=blue,
                     log=True,
                     absolute=False, 
                     cutoff=1e-5,
                     axs=None,
                     fs=20,
                     label=["left vec", "right vec", "matrix"]):
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


def shift_matrix_construction_visualization_subplots(shift_matrix_obj:ShiftMatrix, in_eigenspace=False, log=True, absolute=False, fs=8, cutoff=1e-4):
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



def compose_shift_matrix_construction_visualization(shift_matrix_obj:ShiftMatrix, log=True, absolute=True,fs=8):
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
    -------
    save_dir : str
        Path to the saved combined SVG file (plots\\combined.svg).
    """


    # generate the subfigures based on the properties of shift_matrix_obj
    file_paths_physical=shift_matrix_construction_visualization_subplots(shift_matrix_obj, in_eigenspace=False, log=log, absolute=absolute, fs=fs, cutoff=1e-4)
    file_paths_eigenspace=shift_matrix_construction_visualization_subplots(shift_matrix_obj, in_eigenspace=True, log=log, absolute=absolute, fs=fs, cutoff=1e-4)
    
    #create new SVG figure
    fig = sg.SVGFigure("17cm", "6.5cm")
    #fig.append(sc.Grid(10,10)) # visual grid for ease of aligning figures

    # helper function to loop over, load and add all the generated SVGs to the figure, with appropriate positioning and scaling
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


def plot_dynamics(t_final,y_0,args,t_0=0.,steps=8000):
    """
    Function to plot the numerically integrated state vector y at time t, given the model and perturbation.

    Parameters:
    -------
    t_final: float
        Time until which to integrate.
    y_0: array-like
        Initial value.
    args: tuple
        Tuple of the shape (model_ode, model_args, perturbation, perturbation_args) where:
        - model_ode: function
            The model specific function that computes the time derivative of the state vector based on the model.
        - model_args: tuple
            Arguments for the model specific function.
        - perturbation: function
            The perturbation function that computes the effect of the perturbation on the state vector. If set to None no peturbation will be added.
        - perturbation_args: tuple
            Arguments for the perturbation function.
        - shift: function
            The shift function that computes the effect of the shift on the state vector. If set to None no shift will be added.
        - shift_args: tuple
            Arguments for the shift function.

    Returns:
    -------
    string
        Path to the generated plot.
    """

    t,y=dynamics.integrate_f(t_final,y_0,args,t_0=0.,steps=8000)
    for i in range(int(np.shape(y)[0]/2)):
        plt.plot(t,y[i,:])
    plt.show()


# Example usage
if __name__ == "__main__":
    
    # Create a model and compute the Jacobian
    dim=8
    model = sokm.from_random_sparse_graph(num_nodes=dim, edge_probability=0.3, damping_coefficient=0.01)
    #model.compute_jacobian()
    #model.summary()
    #model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-04-07_11-21-27")

    # Create a ShiftMatrix object
    shift_matrix_obj = ShiftMatrix(model=model)

    # Generate a shift matrix
    eigenvalue_indices = [ 2,3]
    shifts = np.array([-0.6, 0.3])
    zero_rows = np.array([3,4,5,6,7])
    #zero_rows=np.arange(16)

    
    zero_cols = np.array([],dtype=int)
    shift_matrix_obj.construct_from_scratch(eigenvalue_indices, shifts, zero_rows, zero_cols)
    

    delta=1e-1
    offset=np.random.rand(dim)*delta
    offset-=np.sum(offset)/80
    fixpoint=model.compute_fixed_point()
    y_0=np.concatenate((fixpoint,np.zeros(dim)))
    print("y_0 dim: ",len(y_0))
    #eigenvalues, eigenvectors = la.eigh(model.jacobian_matrix)
    #print(eigenvalues)
    node_index=1
    amplitude= delta
    frequency=0.1
    pert_args=amplitude, frequency, node_index 
    pert=(dynamics.cos_perturbation_single_node,pert_args)
    pert=(None,None)
    shift_args= shift_matrix_obj.shift_matrix, fixpoint, offset
    shift=(dynamics.jacobian_shift,shift_args)
    #shift=(None,None)
    plot_dynamics(600,y_0,args=(model.kuramoto_ode,None)+pert+shift)

    #resonance_plot(model,log=True, show_resonance_location=True)
    #pre_and_post_shift_comparison_subplots(model, shift_matrix_obj)
    


    
    #shift_matrix_obj= ShiftMatrix.load_from_file("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-03-31_10-32-17\\shift_matrix.json")

    #compose_shift_matrix_construction_visualization(shift_matrix_obj, log=True, absolute=True)
    