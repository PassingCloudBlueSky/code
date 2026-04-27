"""
This module contains functions to compute the time derivative of the state vector for a given model, 
including the effect of perturbations and shifts. The unperturbed unshifted dynamics are exported from the 
model module, while the perturbations are defined as separate functions in this module to allow for the application across different model dynamics.
Similarly the shift functions are defined separately toallow for the application across different model dynamics.
The main function `f` computes the time derivative 
based on the model's ordinary differential equations (ODEs) and any specified perturbations or shifts. 

"""


import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import ode


""" 
Perturbation functions: 
-------------------------------------------------------------------------------------------------------------------------------
"""

def cos_perturbation_single_node(t,y,args):
    """
    Example perturbation function that computes a cosine perturbation based on the current time and state vector.

    Parameters:
    -------
    t: float
        Current time.
    y: array-like
        Current state vector.
    args: tuple
        Tuple of the shape (amplitude, frequency) where:
        - amplitude: float
            The amplitude of the cosine perturbation.
        - frequency: float
            The frequency of the cosine perturbation.
        - node_index: int
            The index of the state variable to which the perturbation should be applied.

    Returns:
    -------
    perturbation_value: array-like
        The computed cosine perturbation value based on the current time applied to the specified node index in the state vector.
    """

    amplitude, frequency, node_index = args

    # Example perturbation: a simple cosine function of time with given amplitude and frequency
    perturbation_vector = np.zeros_like(y)
    perturbation_vector[node_index] = amplitude * np.cos(2 * np.pi * frequency * t)

    return perturbation_vector


"""
shift functions:
-------------------------------------------------------------------------------------------------------------------------------
"""

def jacobian_shift(t,y,args):
    """
    Applies a shift that effectively adds a shift matrix to the Jacobian of a system.

    Parameters:
    ------------
    t: float
        Current time.
    y: array-like
        Current state vector.
    args: tuple
        Tuple of the shape (shift_matrix, fixpoint) where:
        - shift_matrix: array-like
            The shift matrix to be applied.
        - fixpoint: array-like
            The fixed point around which the shift is applied.
        - offset: array-like
            Optional offset to shift the fixpoint. When provided as zero the fixpoint remains unchanged.
    
    Returns:
    ------------
    shift_value: array-like
        The computed shift value based on the current state vector and the provided shift matrix, ensuring that the fixed point remains unchanged.
    """

    S, fixpoint, offset =args

    y_dot= S @ (y[:np.shape(S)[1]]-fixpoint+offset)

    return np.concatenate(( np.zeros(len(y)-len(y_dot)),y_dot))


def hessian_shift(t,y,args):
    """
    Applies a shift that effectively adds a shift matrix to the Hessian of a system.

    Parameters:
    ------------
    t: float
        Current time.
    y: array-like
        Current state vector.
    args: tuple
        Tuple of the shape (shift_matrix, fixpoint) where:
        - shift_matrix: array-like
            The shift matrix to be applied.
        - fixpoint: array-like
            The fixed point around which the shift is applied.
        - offset: array-like
            Optional offset to shift the fixpoint. When provided as zero the fixpoint remains unchanged.
    
    Returns:
    ------------
    shift_value: array-like
        The computed shift value based on the current state vector and the provided shift matrix, ensuring that the fixed point remains unchanged.
    """

    S, fixpoint, offset =args

    y_dot= S @ np.kron((y[:np.shape(S)[1]]-fixpoint+offset),(y[:np.shape(S)[1]]-fixpoint+offset))

    return np.concatenate(( np.zeros(len(y)-len(y_dot)),y_dot))
 

"""
junction functions:
-------------------------------------------------------------------------------------------------------------------------------

"""


def f(t,y,args):
    """
    Function to compute the time derivative of the state vector y at time t, given the model and perturbation.

    Parameters:
    -------
    t: float
        Current time.
    y: array-like
        Current state vector.
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
    y_dot: array-like
        The time derivative of the state vector y at time t, including the effect of the perturbation if provided.
    """

    model_ode,model_args,perturbation,perturbation_args,shift,shift_args=args
    
    y_dot=model_ode(t,y,model_args)

    # adding the portubation if provided
    if perturbation is not None:
        perturbation_value = perturbation(t,y,perturbation_args)
        #accounting for models of higher order than the perturbation:
        perturbation_value = np.concatenate((perturbation_value, np.zeros(len(y)-len(perturbation_value))))
        
        y_dot+=perturbation_value

    # adding the shift if provided
    if shift is not None:
        shift_value= shift(t,y,shift_args)
        #accounting for models of higher order than the shift:
        y_dot+=np.concatenate((shift_value, np.zeros(len(y)-len(shift_value))))
    
    return y_dot



"""
numerical integrators:
-------------------------------------------------------------------------------------------------------------------------------------------
"""


def integrate_f(t_final,y_0,args,t_0=0.,steps=8000):


    integrator=ode(f).set_integrator("dop853")
    integrator.set_initial_value(y_0,t_0).set_f_params(args)


    dt=(t_final-t_0)/steps

    times=np.zeros(steps+1)
    y_vals=np.zeros((len(y_0),steps+1))
    y_vals[:,0]=y_0

    step=0
    while integrator.successful() and step<steps:
        step+=1
        times[step]=integrator.t+dt
        print("At time {}/{}".format(np.round(times[step],decimals=1),t_final),end="\r")
        y_vals[:,step]=integrator.integrate(integrator.t+dt)
    return times, y_vals