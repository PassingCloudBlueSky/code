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



# Color definitions
blue=np.array([0,170,212,100])/256
darkblue = np.array([0, 68, 170, 100])/256
purple = np.array([205, 135, 222, 100])/256
red = np.array([211, 95, 95, 100])/256
orange = np.array([255, 153, 85, 100])/256
green = np.array([44, 160, 90, 100])/256
black = np.array([0, 0, 0, 100])/256



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
    if num_shifts==1:
        shift_colors.append(list(individual_shift_colors_cmap(0.)))
    else:
        for i in range(num_shifts):
            shift_colors.append(list(individual_shift_colors_cmap (i / (num_shifts - 1))))

    print("shift_colors:", np.shape(shift_colors))
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
                     label=["left vec", "right vec", "matrix","a)"],
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
        if name is not None and axs is None and size[0]>1.4:
            if "eigenspace" in name:
                fig.suptitle(r"eigenspace:",fontsize=fs,color="dimgrey")
            elif "physical" in name:
                fig.suptitle(r"physical space:",fontsize=fs,color="dimgrey")
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


def shift_matrix_construction_visualization_subplots(shift_matrix_obj:ShiftMatrix, in_eigenspace=False, log=True, absolute=False, fs=10, cutoff=1e-4, jac_color=darkblue, node_reference = False, size=(1.5, 1.5)):
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
    n_individual_shift_matrices=len(individual_shift_matrices)
    if in_eigenspace:
        jacobian= np.diag(shift_matrix_obj.eigenvalues)
    else:
        jacobian = shift_matrix_obj.jacobian
    dim=jacobian.shape[0]
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
    if in_eigenspace:
        axes_layered[1,1].set_xlabel(r"$V(J+\sum_{i}\vec{p}_{i}\vec{q}_i^T)V^{-1}$", fontsize=fs)
        panel_label=rf"{chr(97+2*n_individual_shift_matrices+3)})"
        
    else:
        axes_layered[1,1].set_xlabel(r"$J+\sum_{i}\vec{p}_{i}\vec{q}_i^T$", fontsize=fs)
        panel_label=rf"{chr(97+n_individual_shift_matrices+1)})"
    axes_layered[1,1].annotate(rf"$\mathrm{{{panel_label}}}$", (-1.5, 8.5), fontsize=fs, annotation_clip=False)

    # create empty list of saving_paths
    saving_paths=[]

    # Visualize jacobian in eigenspace and physical space
    add_pcolormesh(axes_layered[1,1],jacobian, color=jac_color, log=log, absolute=absolute, cutoff=cutoff, min_max=min_max)

    if in_eigenspace:
        labels=[None,None,r"$VJV^{-1}$",rf"{chr(97+n_individual_shift_matrices+2)})"]
    else:
        labels=[None,None,r"$J$",rf"{chr(97)})"]
    saving_paths.append(visualize_matrix(matrix=jacobian, label=labels, color=jac_color, log=log, absolute=absolute, fs=fs, cutoff=cutoff, name=f"{"eigenspace" if in_eigenspace else "physical"}_jacobian.svg", save_dir=save_dir, min_max=min_max, size=size))
    

    # looping over individual shift matrices to visualize their construction one by one
    
    shift_colors=create_shift_colors(n_individual_shift_matrices) 

    for index in range(n_individual_shift_matrices):
        print(f"Visualizing shift component {index+1} in {'eigen' if in_eigenspace else 'physical'} space...")
        

        # generating labels for subfigures
        p="p"
        q="q"
        if in_eigenspace:
            labels = [rf"$V\vec{{{p}}}_{{{index+1}}}$",
                       rf"$\vec{{{q}}}_{{{index+1}}}^TV^{{{-1}}}$",
                         f"$V\\vec{{{p}}}_{{{index+1}}}\\vec{{{q}}}_{{{index+1}}}^TV^{{{-1}}}$",
                         rf"{chr(100+n_individual_shift_matrices+index)})"]
        else:
            labels = [rf"$\vec{{{p}}}_{{{index+1}}}$", 
                      rf"$\vec{{{q}}}_{{{index+1}}}^T$", 
                      f"$\\vec{{{p}}}_{{{index+1}}}\\vec{{{q}}}_{{{index+1}}}^T$",
                      rf"{chr(98+index)})"]
        add_pcolormesh(axes_layered[1,1], individual_shift_matrices[index], color=shift_colors[index], log=log, absolute=absolute, cutoff=cutoff, min_max=min_max)
        saving_paths.append(visualize_matrix(left_vec=left_generators[index], right_vec=right_generators[index],label=labels, color=shift_colors[index], log=log, fs=fs, cutoff=cutoff, name=f"{"eigenspace" if in_eigenspace else "physical"}_shift_component_{index+1}.svg", save_dir=save_dir, min_max=min_max,size=size))

    # add row numbers to layered plot
    if node_reference and (in_eigenspace is False) and dim <=10:
        
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



def compose_shift_matrix_construction_visualization_horizontal(shift_matrix_obj:ShiftMatrix, log=True, absolute=True,fs=8, jac_color=darkblue, overwrite=False, type="thumbnail"):
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
    save_dir=os.path.join(shift_matrix_obj.current_dir,f"combined_horizontal_{type}.svg")
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

    # generate the subfigures based on the properties of shift_matrix_obj
    file_paths_physical=shift_matrix_construction_visualization_subplots(shift_matrix_obj, in_eigenspace=False, log=log, absolute=absolute, fs=fs, cutoff=1e-4, jac_color=jac_color, size=size)
    file_paths_eigenspace=shift_matrix_construction_visualization_subplots(shift_matrix_obj, in_eigenspace=True, log=log, absolute=absolute, fs=fs, cutoff=1e-4, jac_color=jac_color, size=size)
    
    
    #combined_fig.append(sc.Grid(20,20)) # visual grid for ease of aligning figures

    # helper function to loop over, load and add all the generated SVGs to the figure, with appropriate positioning and scaling
    def add_svg_row_to_figure(fig,file_paths, second_row=False, scale=80):
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
        y_0 = scale*1.05 if second_row else 0

        # list to store "+"/"=" sign figures such that they can be appended to fi in the end and are layered on top of the other svgs, are not covered by them
        symbols=[]

        #looping over the individual SVGs
        x_0=0
        for i,path in enumerate(file_paths):

            # horizontal offset of objects to append
            if i>0:
                x_0 = i*scale - 10
            if i==len(file_paths)-1:
                x_0 = i*scale  

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
            
            if False:
                # generating and appending +/= signs to illustrate the narrative between the figures
                if i<len(file_paths)-2:
                    plus=sg.TextElement(x_0+65,y_0+32,"+",size=6)
                    symbols.append(plus)
                elif i==len(file_paths)-2:
                    equal=sg.TextElement(x_0+65,y_0+32,"=",size=6)
                    symbols.append(equal)
                

                
    add_svg_row_to_figure(combined_fig, file_paths_eigenspace, second_row=True,scale=scale)
    add_svg_row_to_figure(combined_fig, file_paths_physical, second_row=False,scale=scale)
    # dividing line between the spaces
    line=sc.Line([(0,scale*1.1),(scale*(len(file_paths_physical))-5,scale*1.1)],width=0.8)
    combined_fig.append([line])

    

    combined_fig.save(save_dir)
    png_path=os.path.join(shift_matrix_obj.current_dir,f"combined_horizontal_{type}.png")
    svg2png(url=save_dir,write_to=png_path,
            parent_height=scale*2.2,parent_width=scale*(len(file_paths_eigenspace)-0.05),output_height=scale*2.2,output_width=scale*(len(file_paths_eigenspace)-0.05))
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

def plot_dynamics(t_final,
                    y_0,
                    args,
                    axes=None,
                    t_0=0.,
                    steps=8000,
                    name: Optional[str]=None,
                    save_dir: Optional[str]=None,
                    node_colors=None):
    """
    Function to plot the numerically integrated state vector y at time t, given the model and perturbation.

    Parameters:
    -------
    t_final: float
        Time until which to integrate.
    y_0: array-like
        Initial value.
    args: tuple
        Tuple of the shape (model_ode, model_args, perturbation, perturbation_args, shift, shift_args) where:
        - model_ode: function
            The model specific function that computes the time derivative of the state vector based on the model.
        - model_args: tuple
            Arguments for the model specific function.
        - perturbation: function or None
            The perturbation function that computes the effect of the perturbation on the state vector.
            If set to None no perturbation is added.
        - perturbation_args: tuple or None
            Arguments for the perturbation function.
        - shift: function or None
            The shift function that computes the effect of the shift on the state vector.
            If set to None no shift is added.
        - shift_args: tuple or None
            Arguments for the shift function.
    axes: matplotlib.axes._axes.Axes, optional
        Axis to draw the dynamics on. If None, a new figure is created.
    name: str, optional
        File name when saving the plot if axes is not provided.
    save_dir: str, optional
        Directory to save the figure when axes is not provided.

    Returns:
    -------
    Optional[str]
        Path to the generated plot when a new figure was created, otherwise None.
    """
    created_fig = False
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(2.1, 0.7))
        created_fig = True

    t, y = dynamics.integrate_f(t_final, y_0, args, t_0=t_0, steps=steps)
    for i in range(int(np.shape(y)[0] / 2)):
        axes.plot(t, y[i, :],color=node_colors[i] if node_colors is not None else None, linewidth=0.5)

    # aesthetics and labels

    if created_fig:
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "unorganized_plots")
        return save_figure(fig, save_dir=save_dir, name=name if name is not None else "dynamics.svg")




def plot_dynamics_scenario(t_final: float, args, model: BaseModel, y_0=None, steps=8000, meta_scenario_name="default",
                    save_dir: Optional[str]=None, ax=None, node_colors=None):
    """
    Generates two panels comparing the dynamics of the network with and without vtn for a given scenario.

    If axes are provided, only the unshifted plot is drawn to axes[0] and the shifted plot
    to axes[1]. If no perturbation is provided, only the offset transient comparison is created.
    """

    # unpacking arguments
    instance = (model.model_ode, model.model_ode_args)
    pert = args[0:2]
    shift = args[2:4]
    no_perturbation = (pert[0] is None)

    dim = np.shape(model.jacobian_matrix)[0]

    # set y_0 to fixpoint wiht an offset if no perturbation is provided or perturbation_and_offset is True, unless y_0 is provided
    if y_0 is None:
        if model.fixed_point is None:
            model.compute_fixed_point()
        fixpoint = model.fixed_point
        y_0 = np.zeros(model.ode_dimension)
        y_0[:dim] = fixpoint
    
        # creating offset if no perturbation is provided
        if pert[0] is None:
            offset = np.random.rand(dim) * pert[1]
            offset -= np.sum(offset) / dim
            y_0[:dim] += offset

    # creating figure if no axes have been provided for plotting
    if ax is None:
        fig,ax=plt.subplots(2,1,figsize=(3, 2), sharex=True)

        # setting save_dir to current instance directory not provided as a parameter to save the generated figure
        if save_dir is None:
            if model.current_dir is None:
                model.save_parameters()
            save_dir=model.current_dir
            save_dir=os.path.join(save_dir,"plots") 
    else:
        save_dir=None

    # setting up shifted and unshifted cases for plotting
    cases=[("unaltered", instance+pert+(None,None)),
           ("shifted", instance+pert+shift)]
    
    # looping over cases and plotting dynamics for each case in the respective subplot
    for i, (name, args) in enumerate(cases):
        print("Numerically integrating case:", name)
        y_0_case = y_0.copy()
        plot_dynamics(t_final, y_0_case, args,
                    axes=ax[i],
                    t_0=0.,
                    steps=steps,node_colors=node_colors)

    # saving figure if it was created in this function, otherwise just returning the axes for further use
    if save_dir is not None:
        print(save_dir)
        svg_path=save_figure(fig, save_dir=save_dir, name=meta_scenario_name+"_dynamics.svg")
        png_path=os.path.join(save_dir,meta_scenario_name+"_dynamics.png")
        svg2png(url=svg_path,write_to=png_path,
                parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)
        return svg_path
    else:
        return ax
    

def plot_scenario_dynamics_split(t_final: float, args, model: BaseModel, nodes: list, y_0=None, steps=8000, meta_scenario_name="default",
                    save_dir: Optional[str]=None, ax=None):
    """
    Generates two panels comparing the dynamics of the network with and without vtn for a given scenario.

    If axes are provided, only the unshifted plot is drawn to axes[0] and the shifted plot
    to axes[1]. If no perturbation is provided, only the offset transient comparison is created.
    """

    # unpacking arguments
    instance = (model.model_ode, model.model_ode_args)
    pert = args[0:2]
    shift = args[2:4]

    dim = np.shape(model.jacobian_matrix)[0]

    # set y_0 to fixpoint wiht an offset if no perturbation is provided or perturbation_and_offset is True, unless y_0 is provided
    if y_0 is None:
        if model.fixed_point is None:
            model.compute_fixed_point()
        fixpoint = model.fixed_point
        y_0 = np.zeros(model.ode_dimension)
        y_0[:dim] = fixpoint

    # creating figure if no axes have been provided for plotting
    if ax is None:
        fig,ax=plt.subplots(len(nodes),1,figsize=(3, 2), sharex=True)

        # setting save_dir to current instance directory not provided as a parameter to save the generated figure
        if save_dir is None:
            if model.current_dir is None:
                model.save_parameters()
            save_dir=model.current_dir
            save_dir=os.path.join(save_dir,"plots") 
    else:
        save_dir=None

    # numerically integrating shifted and unshifted trajectories
    t, ys = dynamics.integrate_f(t_final, y_0, instance+pert+(None,None), t_0=0, steps=steps)
    t, ys_shifted = dynamics.integrate_f(t_final, y_0, instance+pert+shift, t_0=0, steps=steps)

    # fix point deviation
    ys[:len(y_0)] -= y_0[:, None]
    ys_shifted[:len(y_0)] -= y_0[:, None]

    filepath = os.path.join(shift[1][0].current_dir,meta_scenario_name)
    np.savez(filepath, t=t, ys=ys, ys_shifted=ys_shifted)
    traj = np.load(filepath+".npz", allow_pickle=True)
    t_l=traj["t"]
    ys_l=traj["ys"]
    ys_shifted_l=traj["ys_shifted"]
    if np.any(t_l != t) or np.any(ys!=ys_l) or np.any(ys_shifted_l != ys_shifted):
        raise ValueError("Loaded trajectories don't equal saved trajectories!")

    # min and max for consistent scaling across subplots
    min_max= np.empty((2, len(nodes)))

    # looping over nodes, plotting shifted and unshifted trajectory for each node into the respective subplot
    for i, node in enumerate(nodes):
        min= np.min(np.stack((ys[node, :], ys_shifted[node, :])))
        max= np.max(np.stack((ys[node, :], ys_shifted[node, :])))
        min_max[:, i] = (min,max)
        ax[i].plot(t, ys[node, :], color=black, linewidth=0.5, label="unshifted")
        ax[i].plot(t, ys_shifted[node, :], color=red, linewidth=0.5, label="shifted")
        if i < len(nodes)-1:
            ax[i].set_xticks([])


    # saving figure if it was created in this function, otherwise just returning the axes for further use
    if save_dir is not None:
        print(save_dir)
        svg_path=save_figure(fig, save_dir=save_dir, name=meta_scenario_name+"_dynamics.svg")
        png_path=os.path.join(save_dir,meta_scenario_name+"_dynamics.png")
        svg2png(url=svg_path,write_to=png_path,
                parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)
        return svg_path
    else:
        return min_max
    

def plot_scenario(t_final: float, args, model: BaseModel, nodes: list, ax, y_0=None, steps=8000, meta_scenario_name="default",
                    save_dir: Optional[str]=None, node_colors=None):
    """
    Plots for a given scenario the psd, perturbation trajectory and trajectories of the nodes w and w/o shift in a row of subplots. 
    The psd is plotted in ax[0], the perturbation trajectory in ax[1] and the trajectories of the nodes w and w/o shift in ax[2] to ax[2+len(nodes)].
    """

    # unpacking
    pert = args[0:2]
    y_lims= np.empty((2, 2+len(nodes)))

    # plotting hist of psd of perturbation into ax[0]
    y_lims[:,0]=plot_scenario_psd(ax=ax[0], pert=pert, log=True)

     # plotting trajectory of perturbation into ax[1]
    y_lims[:,1]=plot_scenario_perturbation_trajectory(ax=ax[1], t_final=t_final, steps=steps, pert=pert)

     # plotting trajectories of nodes w and w/o shift into ax[2] to ax[2+len(nodes)]
    y_lims[:, 2:2+len(nodes)]=plot_scenario_dynamics_split(t_final, args, model, nodes, y_0=y_0, steps=steps, meta_scenario_name=meta_scenario_name,
                                     ax=ax[2:2+len(nodes)])

    return y_lims


def plot_scenario_psd(ax, pert, f_min_max=(0.5, 3.5), log= True):
    """
    Generates a power spectrum density plot of the perturbation for a given scenario. 
    """
    # unpacking perturbation type and the corresponding arguments
    perturbation_type=pert[0]
    perturbation_args=pert[1]
    min_max=np.array([0,0])

    # checking for known perturbation types and plotting the corresponding psd into ax
    if perturbation_type is None:
        raise ValueError("No perturbation provided for scenario, cannot plot psd.")
    
    elif perturbation_type is dynamics.perturbation_from_psd:
        noise_object=perturbation_args[0]
        freqs = noise_object.freqs
        psd= noise_object.psd
        ax.plot(freqs, psd)
        min_max=np.array([np.min(psd), np.max(psd)])
        
    elif perturbation_type is dynamics.sine_perturbation_single_node:
        amplitude=perturbation_args[0]
        frequency=perturbation_args[1]
        freqs=np.array([frequency])
        amps=np.array([amplitude])
        ax.hist(freqs, amps, markersize=5)
        ax.set_xlim(f_min_max)
        min_max=np.array([0, amplitude*1.5])

    if log:
        ax.set_xscale("log")
    
    return min_max
    


def plot_scenario_perturbation_trajectory(ax, t_final, steps, pert):
    """
    Plots the trajectory of the perturbation over time for a given scenario into ax.
    """
    # generate time array
    t = np.linspace(0, t_final, steps)

    # unpacking perturbation type and the corresponding arguments
    perturbation_func=pert[0]
    perturbation_args=pert[1]
    perturbed_node=perturbation_args[-1]

    
    # generating perturbation values from the time array
    vals=perturbation_func(t, np.empty(perturbed_node+1), perturbation_args)[perturbed_node]

    # plotting the perturbation trajectory into ax
    ax.plot(t, vals)
    return np.array([np.min(vals), np.max(vals)])


def plot_scenario_comparison_split(t_final,
                               model:sokm,
                               shift_matrix_obj:ShiftMatrix,
                               scenarios,
                               nodes=[0,3,5],
                               steps=8000,
                               y_0=None,
                               name="dynamics_comparison", 
                               save_dir=None,
                               fontsize=5,
                               perturbed_node=1):


    """
    


    Arguments:
        scenarios: list of touples the tuples have the form (title, pert) where 
            title: string
                title of the scenario.
            pert: tuple
                of the form (perturbation, perturbation_args) that defines the perturbation for the scenario
         """

    # initializing figure
    fig,ax=plt.subplots(2+len(nodes), len(scenarios),figsize=(7, 3), sharey="row", gridspec_kw=dict(hspace=0.1, wspace=0.1))
    
    # colorshemes
    cmap=plt.cm.cividis.reversed()
    node_colors = node_colors_from_perturbation_distance(perturbed_node, model, cmap=cmap)

    # Construct shift arguments
    shift_args = (shift_matrix_obj, model, None)
    shift = (dynamics.jacobian_shift, shift_args)
        
    y_lims=np.empty((2, 2+len(nodes))) # for consistent scaling across subplots, storing min and max values of each subplot for each scenario to later set the same y limits for the respective subplots across scenarios
    # looping over the scenarios to generate the corresponding plots
    for i, (title, pert ) in enumerate(scenarios):
        print(f"Generating plot for scenario: {title}")

        # Call plot_dynamics_scenarios with these perturbation and shift arguments to plot the row comaptison for the scenario onto the existing axes
        y_lims_dummy = plot_scenario(
            t_final=t_final,
            args=pert + shift,
            nodes=nodes,
            model=model,
            y_0=y_0,
            steps=steps,
            meta_scenario_name=title,
            ax=ax[:, i],
            node_colors=node_colors
        )

        # updating y_lims if necessary
        y_lims[0,:] = np.min(np.stack((y_lims[0,:], y_lims_dummy[0,:])), axis=0)
        y_lims[1,:] = np.max(np.stack((y_lims[1,:], y_lims_dummy[1,:])), axis=0)

        # managing x and y axis ticks and labels for aesthetics
        ax[0,i].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        ax[0,i].set_title(title, fontsize=fontsize)
        ax[1,i].set_xlabel(r"$t$", fontsize=fontsize)
        ax[0,i].set_xlim((0,t_final))
        ax[1,i].set_xlim((0,t_final))
        if i>0: #turning of y axis ticks and labels for the right two columns of plots
            ax[1,i].tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False)
            ax[0,i].tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False)

    ax[0,0].set_ylabel(r"$\theta_n(t)$", fontsize=fontsize)
    #ax[0,0].tick_params(axis='y', labelsize=fontsize)
    #ax[1,0].tick_params(axis='y', labelsize=fontsize)
    ax[1,0].set_ylabel(r"$\theta_{n,S}(t)$", fontsize=fontsize)
    

    # saving the composed figure as SVG and PNG
    if save_dir is None:
        save_dir = shift_matrix_obj.current_dir
    svg_path=save_figure(fig, save_dir=save_dir, name=name+".svg")
    png_path=os.path.join(save_dir,name+".png")
    svg2png(url=svg_path,write_to=png_path,
                parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)
    return svg_path



def plot_scenario_comparison_bunched(t_final,
                               model:sokm,
                               shift_matrix_obj:ShiftMatrix,
                               scenarios,
                               steps=8000,
                               y_0=None,
                               name="dynamics_comparison", 
                               save_dir=None,
                               fontsize=5,
                               perturbed_node=1):


    """
    


    Arguments:
        scenarios: list of touples the tuples have the form (title, pert) where 
            title: string
                title of the scenario.
            pert: tuple
                of the form (perturbation, perturbation_args) that defines the perturbation for the scenario
         """

    # initializing figure
    fig,ax=plt.subplots(2, len(scenarios),figsize=(7, 3), sharey="row", gridspec_kw=dict(hspace=0.1, wspace=0.1))
    
    # colorshemes
    cmap=plt.cm.cividis.reversed()
    node_colors = node_colors_from_perturbation_distance(perturbed_node, model, cmap=cmap)

    # Construct shift arguments
    shift_args = (shift_matrix_obj, model, None)
    shift = (dynamics.jacobian_shift, shift_args)
        

    # looping over the scenarios to generate the corresponding plots
    for i, (title, pert ) in enumerate(scenarios):
        print(f"Generating plot for scenario: {title}")

        # Call plot_dynamics_scenarios with these perturbation and shift arguments to plot the row comaptison for the scenario onto the existing axes
        ax[:,i] = plot_dynamics_scenario(
            t_final=t_final,
            args=pert + shift,
            model=model,
            y_0=y_0,
            steps=steps,
            meta_scenario_name=title,
            ax=ax[:, i],
            node_colors=node_colors
        )

        # managing x and y axis ticks and labels for aesthetics
        ax[0,i].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        ax[0,i].set_title(title, fontsize=fontsize)
        ax[1,i].set_xlabel(r"$t$", fontsize=fontsize)
        ax[0,i].set_xlim((0,t_final))
        ax[1,i].set_xlim((0,t_final))
        if i>0: #turning of y axis ticks and labels for the right two columns of plots
            ax[1,i].tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False)
            ax[0,i].tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False)

    ax[0,0].set_ylabel(r"$\theta_n(t)$", fontsize=fontsize)
    #ax[0,0].tick_params(axis='y', labelsize=fontsize)
    #ax[1,0].tick_params(axis='y', labelsize=fontsize)
    ax[1,0].set_ylabel(r"$\theta_{n,S}(t)$", fontsize=fontsize)
    

    # saving the composed figure as SVG and PNG
    if save_dir is None:
        save_dir = shift_matrix_obj.current_dir
    svg_path=save_figure(fig, save_dir=save_dir, name=name+".svg")
    png_path=os.path.join(save_dir,name+".png")
    svg2png(url=svg_path,write_to=png_path,
                parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)
    return svg_path




def plot_noise(noise_object: dynamics.ContinuousSpectrumNoise, shift_matrix_obj:ShiftMatrix, axes=None, name="noise", save_dir=None):


    if axes is None:
        fig, ax = plt.subplots(1,1)
    else:
        ax = axes
    times = np.arange(0, noise_object.t_max,noise_object.t_max/noise_object.steps)
    noise = noise_object(times)
    print("times", times)
    print(type(noise), noise)
    print("noise at some time", noise_object(times[0]))
    print(noise_object.__dict__)
    ax.plot(times, noise_object(times))

    if axes is None:   
        noise_integrated = np.cumsum(noise)
        ax.plot(times, noise_integrated)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"power")
        ax.set_title(name)
        plt.show()

        if save_dir is None:
            save_dir = shift_matrix_obj.current_dir
        svg_path=save_figure(fig, save_dir=save_dir, name=name+".svg")
        png_path=os.path.join(save_dir,name+".png")
        svg2png(url=svg_path,write_to=png_path,
                parent_height=110,parent_width=400,output_height=115*4,output_width=400*4)


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


#----------------------
# plotting utils for publication figure on dynamics


def plot_scenario_row(t_final, 
                      args, 
                      model, 
                      y_0, 
                      axes, 
                      nodes=[2,3], 
                      steps=8000, 
                      meta_scenario_name="scenario",
                      overwrite=False, 
                      freqs_lims=(0.5, 3.5),
                      min_max_psd=None, 
                      psd_vlines=None,
                      psd_vline_colors=None,
                      alpha=1.):
    """
    Generates scenario data if it can't be loaded. Then adds shift power, power spectral density and selected trajectories to the provided axes. 
    
    """

    # load data if path provided, else generate anew
    traj=generate_or_fetch_scenario_data(t_final, args, model, y_0=y_0, steps=steps, meta_scenario_name=meta_scenario_name, overwrite=overwrite)

    # plot psd histogram of response with illustrated shifts
    freqs, psd = compute_psd_from_traj(traj["t"], traj["ys"][:np.shape(model.jacobian_matrix)[0], :])
    psd=np.mean(psd, axis=0)
    freqs_vtn, psd_vtn = compute_psd_from_traj(traj["t"], traj["ys_shifted"][:np.shape(model.jacobian_matrix)[0], :])
    psd_vtn=np.mean(psd_vtn, axis=0)
    min_max_psd = plot_psd(axes[3],freqs,psd,freqs_vtn=freqs_vtn,psd_vtn=psd_vtn,freqs_lims=freqs_lims, min_max=min_max_psd, vlines=psd_vlines, vline_colors=psd_vline_colors)

    # plot shift powers over time
    axes[0]= plot_vtn_powers(axes[0],traj, args[3][0],model)

    # plot dynamics of selected nodes
    plot_trajectories_split(traj, axes=axes[1:1+len(nodes)], nodes=nodes, y_0=y_0, alpha=alpha)
    return min_max_psd


def compute_psd_from_traj(t, vals, steady_state_t=500):
    """
    Computes the power spectral density from a trajectory using scipy.signal.welch.
    """
    steady_idx=np.argmin(np.abs(t-steady_state_t))
    t=t[steady_idx:]
    vals=vals[:,steady_idx:]
    freqs, psd = scipy.signal.welch(vals, fs=1/(t[1]-t[0]), axis=-1, nperseg=len(t))
    return freqs, psd

def plot_trajectories_split(traj, axes, nodes, y_0=None, alpha=1.):

    t=traj["t"]
    ys=traj["ys"]
    ys_shifted=traj["ys_shifted"]

    if y_0 is not None:
        ys[:len(y_0)] -= y_0[:, None]
        ys_shifted[:len(y_0)] -= y_0[:, None]

     # min and max for consistent scaling across subplots
    for i, node in enumerate(nodes):
        min= np.min(np.stack((ys[node, :], ys_shifted[node, :])))
        max= np.max(np.stack((ys[node, :], ys_shifted[node, :])))
        if i==0:
            min_max= np.array([min,max])
        else:
            if min<min_max[0]:
                min_max[0]=min
            if max>min_max[1]:
                min_max[1]=max
        axes[i].plot(t, ys[node, :], color="black", linewidth=0.5, label="unshifted", alpha=alpha)
        axes[i].plot(t, ys_shifted[node, :], color=darkblue, linewidth=0.5, label="shifted", alpha=alpha)
        if i < len(nodes)-1:
            axes[i].set_xticks([])
    return min_max


def generate_or_fetch_scenario_data(t_final: float, args, model: BaseModel, y_0=None, steps=8000, meta_scenario_name="default",
                    save_dir: Optional[str]=None, overwrite=False, minus_fixpoint=False):

    """
    Checks if the trajectory data of the scenario (w and w/o) VTN already exists. If so it is loaded. Else generated and saved. 
    """

    # unpacking arguments
    instance = (model.model_ode, model.model_ode_args)
    pert = args[0:2]
    shift = args[2:4]

    # checking if data already exists
    if save_dir == None:
        save_dir=shift[1][0].current_dir
    filepath = os.path.join(save_dir,meta_scenario_name)+"_tfinal"+str(t_final)+".npz"

    

    if overwrite is False and os.path.isfile(filepath):
        print(f"Loading trajectory data from {filepath}")
        traj = np.load(filepath, allow_pickle=True)
        return traj
    
    print(f"Simulating trajectory data for {filepath}, since none was found.")

    
    dim = np.shape(model.jacobian_matrix)[0]

    # set y_0 to fixpoint if not provided
    if y_0 is None:
        if model.fixed_point is None:
            model.compute_fixed_point()
        fixpoint = model.fixed_point
        y_0 = np.zeros(model.ode_dimension)
        y_0[:dim] = fixpoint

    # numerically integrating shifted and unshifted trajectories
    t, ys = dynamics.integrate_f(t_final, y_0, instance+pert+(None,None), t_0=0, steps=steps)
    t, ys_shifted = dynamics.integrate_f(t_final, y_0, instance+pert+shift, t_0=0, steps=steps)
    #plt.plot(t,ys_shifted)
    #plt.show()
    if False:
        # fix point deviation
        ys[:len(y_0)] -= y_0[:, None]
        ys_shifted[:len(y_0)] -= y_0[:, None]
    
    np.savez(filepath, t=t, ys=ys-y_0[:,None] if minus_fixpoint else ys, ys_shifted=ys_shifted- y_0[:,None] if minus_fixpoint else ys_shifted)
    
    return np.load(filepath, allow_pickle=True)


def plot_psd(axes,freqs,psd,freqs_vtn=None,psd_vtn=None,freqs_lims=None, min_max=None, vlines=None, vline_colors=None):
    """
    Plots a histogram for the provided power spectral density. If psd_vtn is provided it is layered on top.
    """
    # updating min_max
    if min_max is not None:
        min_max[0]=min(min_max[0], np.min(psd),np.min(psd_vtn) if psd_vtn is not None else np.inf)
        min_max[1]=max(min_max[1], np.max(psd),np.max(psd_vtn) if psd_vtn is not None else -np.inf)
    else:
        min_max=np.empty(2)
        min_max[0]=min( np.min(psd),np.min(psd_vtn) if psd_vtn is not None else np.inf)
        min_max[1]=max( np.max(psd),np.max(psd_vtn) if psd_vtn is not None else -np.inf)

    # plotting psd hist
    axes.plot(freqs, psd, color="grey",alpha=0.8)
    #axes.hist(psd, bins=freqs, color="grey", alpha=0.8)
    
    # plotting psd_vtn if provided
    if np.any(psd_vtn!=None) and np.any(freqs_vtn!=None):
        axes.plot(freqs_vtn, psd_vtn, color=darkblue, alpha=0.8)
        #axes.hist(psd_vtn, bins=freqs_vtn, color="grey", alpha=0.8)

    # plotting vertical lines if provided
    if vlines is not None and vline_colors is not None:
        for vline, color in zip(vlines, vline_colors):
            axes.axvline(x=vline/(2*np.pi), color=color, linestyle="--", linewidth=0.5, alpha=1.)

    if freqs_lims is not None:
        axes.set_xlim(freqs_lims)
    #axes.set_xscale("log")
    axes.set_ylim(bottom=1e-8)
    axes.set_yscale("log")
    axes.set_xlabel(r"$\omega/2\pi$ [Hz]")

    return min_max

    
def plot_vtn_powers(axes, traj, shift_matrix_obj:ShiftMatrix, model: sokm, threshhold= 1e-8, absolute=False):
    """
    calculates the shift power of the nodes with battery energy storage systems
    """

    # fetching indices nodes where powers are applied based on the shift matrix
    vtn_nodes= np.nonzero(np.any(shift_matrix_obj.shift_matrix > threshhold, axis=1))[0]

    # fetching trajectory data and fixpoint values
    t=traj["t"]
    y_fixpoint=model.compute_fixed_point()
    ys_shifted=traj["ys_shifted"][:len(y_fixpoint),:]

    # compute shift power from that
    p_vtn= (shift_matrix_obj.shift_matrix@(ys_shifted-y_fixpoint[:,None]))[vtn_nodes,:]
    p_vtn_average= np.cumsum(p_vtn,axis=1)[:,1:]/t[None,1:] #skipping first entry in order to avoid division by zero

    if absolute:
        p_vtn_average=np.abs(p_vtn_average)
    # bess colors
    #colors=["orange", "blue","green","red","darkblue"]
    colors=[]
    # plot
    for bess_index in range(len(vtn_nodes)):
        #axes.plot(t,p_vtn[bess_index], color=colors[bess_index],linewidth=0.5)
        axes.plot(t[1:],p_vtn_average[bess_index],color=colors[bess_index] if len(colors) > bess_index else orange,linewidth=0.5, alpha=0.8)
        #plt.plot(t,p_vtn[bess_index], color=colors[bess_index])
        #plt.plot(t[1:],p_vtn_average[bess_index], linestyle="--",color=colors[bess_index])
        #plt.show()
    if absolute:
        axes.set_yscale("log")
    axes.set_xlabel(r"$t $ [s]")
    axes.set_ylabel(r"$\overline{P}_{VTN}$")
    
    return axes
        

# Example usage
if __name__ == "__main__":
    
    model=sokm.load_from_folder("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Moritz_vals")

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
    print("type is Shift matrix:", isinstance(shift_matrix_obj, ShiftMatrix))
    #resonance_plot(model,log=True, show_resonance_location=True)
    #pre_and_post_shift_comparison_subplots(model, shift_matrix_obj)
    compose_shift_matrix_construction_visualization_vertical(shift_matrix_obj, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=True)
    compose_shift_matrix_construction_visualization_horizontal(shift_matrix_obj, log=True, absolute=True,fs=10, jac_color=darkblue,overwrite=True,type="shift")
    


    
    #shift_matrix_obj= ShiftMatrix.load_from_file("C:\\Users\\leand\\Documents\\Ausprobieren\\TU Dresden WHK\\code\\data\\SecondOrderKuramotoModel\\Instance_2026-03-31_10-32-17\\shift_matrix.json")

    #compose_shift_matrix_construction_visualization(shift_matrix_obj, log=True, absolute=True)
    