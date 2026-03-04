from IPython.display import SVG
import matplotlib.pyplot as plt
import numpy as np
from kuramoto_class import SecondOrderKuramotoModel as sokm 
import scipy
import networkx as nx
import os
import datetime

def save_figure(fig, model: sokm, name: str) -> str:

        """
        Save the figure 

        
        """

        # ascertaining existence of model instance directory
        if model.current_dir == None:
            model.save_parameters()
        save_dir = os.path.join(model.current_dir, "plots")
        os.makedirs(save_dir, exist_ok=True)

        # creating plot path/name
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
        save_path=os.path.join(save_dir, date_str+name)

        fig.savefig(save_path,bbox_inches='tight')
        
        return save_dir


def resonance_plot(model: sokm,a=0.01,log=False, extra_scope=0.2, show_resonance_location=False, fs=15):
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
        plt.plot(omega,response_vals[i,:],alpha=0.15,color="royalblue")
        
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
        ax.set_ylim(bottom=0.1*np.min(response_vals))
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

    save_figure(fig, model, name="resonance_plot")
    plt.show()



def plot_network(model: sokm):
    """
    
    """

    G = nx.from_numpy_array(model.connectivity_matrix)
    fig, ax = plt.subplots(figsize=(14,14))
    nx.draw(G)
    save_figure(fig, model, name="network")
    plt.show()


# Example usage
if __name__ == "__main__":
    model = sokm.from_random_sparse_graph(num_nodes=8, edge_probability=0.5, damping_coefficient=0.01, seed=42)
    model.compute_jacobian()
    model.summary()
    plot_network(model)
    resonance_plot(model,log=True, show_resonance_location=True)