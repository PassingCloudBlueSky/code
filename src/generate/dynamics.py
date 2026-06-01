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
import matplotlib.pyplot as plt
import os


""" 
Perturbation functions: 
-------------------------------------------------------------------------------------------------------------------------------
"""


if False:
    def sine_perturbation_single_node(t,y,args):
        """
        Example perturbation function that computes a sine perturbation based on the current time and state vector.

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
        if np.isscalar(t):
            perturbation_vector = np.zeros_like(y)
            perturbation_vector[node_index] = amplitude * np.sin( frequency * t)
        else:
            perturbation_vector = np.zeros((len(y),len(t)))
            perturbation_vector[node_index,:] = amplitude * np.sin( frequency * t)
        # TODO: loop over node_indices if node_index is array of int
        

        return perturbation_vector


    def perturbation_from_noise(t,y,args):

        times , noise = args

        current_index=np.searchsorted(times,t)
        perturbation_vector = np.zeros_like(y)
        perturbation_vector[:len(noise)]=noise[current_index]

        return perturbation_vector



    def spectrum_noise(spectrum_func, func_params, max_val=0.05, steps=1024, samples_per_step=10, rate=1.):
        """
        Generates noise based on a given spectrum function. 
        The spectrum function should take an array of frequencies as input 
        and return the corresponding power spectral density values.
        Parameters:
        -------------
        spectrum_func: function
            A function that takes an array of frequencies and returns the corresponding power spectral density values.
        max_val: float
            The maximum amplitude of the generated noise.
        steps: int
            The number of steps to generate.
        samples_per_step: int
            The number of samples per step.
        rate: float
            The sampling rate (samples per unit time). So the total time duration t_max=samples/rate.
        """
        steps +=1 # to ensure we have the correct number of samples after inverse FFT
        samples=steps*samples_per_step
        freqs = np.fft.rfftfreq(samples, 1.0/rate)
        psd = spectrum_func(freqs, **func_params)
        amplitudes = np.sqrt(psd)
        phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))
        spectrum = amplitudes * phases
        noise = np.fft.irfft(spectrum, samples)[np.arange(steps)*samples_per_step]
        integrated_noise = np.cumsum(noise)
        #scaling = max_val/np.max(np.abs(integrated_noise))  # Normalize the integrated noise to the desired maximum amplitude to counteract random walk
        scaling = max_val/np.max(np.abs(noise)) 
        return freqs, psd * scaling, (noise - integrated_noise[-1]/len(integrated_noise))* scaling


class ContinuousSpectrumNoise: 
    def __init__(self, spectrum_func, func_params, max_val=0.05, steps=1024, samples_per_step=1, rate=1., perturbed_node=0):
        """
        Continuous noise generator using sum-of-sinusoids.
        """
        self.steps = steps + 1
        self.samples = self.steps * samples_per_step
        self.rate = rate
        self.t_max = self.steps / rate
        self.freqs = np.fft.rfftfreq(self.samples, 1.0 / rate)
        self.psd = spectrum_func(self.freqs, **func_params)
        self.amps = np.sqrt(self.psd)
        self.phases = 2 * np.pi * np.random.rand(len(self.freqs))
        self.perturbed_node= perturbed_node

        # Scaling the based on a sample of the resulting noise in order to match the provided max_val
        test_t = np.linspace(0, self.t_max, self.samples)
        test_noise = self._sum_of_sinusoids(test_t)
        scaling = max_val / np.max(np.abs(test_noise))
        self.psd*=scaling**2
        self.amps *= scaling

        # throw away all frequencies out whose amps value is below a certain threshold to speed up the noise generation while keeping the same max_val
        # e.g. 1e-2 of the max amp value
        amp_threshold = 1e-2 * np.max(self.amps)
        self.amps[self.amps < amp_threshold] = 0
        self.psd[self.psd < amp_threshold**2] = 0

    def _sum_of_sinusoids(self, t):
        # Broadcasting: shape (len(freqs), len(t))
        return np.sum(
            self.amps[:, None] * np.cos(2 * np.pi * self.freqs[:, None] * t [None,:]+ self.phases[:, None]),
            axis=0
        )

    def __call__(self, t):
        t = np.atleast_1d(t)
        noise = self._sum_of_sinusoids(t)
        # Remove DC drift (mean subtraction)
        # noise -= np.mean(noise)

        return noise if len(noise) > 1 else noise[0]

    def save(self, file_directory, name="noise_obj.npz"):
        """Save the noise generator instance to a file."""
        filepath = os.path.join(file_directory,name)
        np.savez(
            filepath,
            rate=self.rate,
            steps=self.steps,
            samples=self.samples,
            t_max=self.t_max,
            freqs=self.freqs,
            psd=self.psd,
            amps=self.amps,
            phases=self.phases,
            perturbed_node=self.perturbed_node,
        )
        return filepath

    @classmethod
    def from_file(cls, filepath):
        """Load a ContinuousSpectrumNoise generator from a saved file."""
        data = np.load(filepath, allow_pickle=True)
        obj = cls.__new__(cls)
        obj.rate = float(data["rate"])
        obj.steps = int(data["steps"])
        obj.samples = int(data["samples"])
        obj.t_max = float(data["t_max"])
        obj.freqs = data["freqs"]
        obj.psd = data["psd"]
        obj.amps = data["amps"]
        obj.phases = data["phases"]
        obj.perturbed_node = int(data["perturbed_node"])
        return obj


def perturbation_from_psd(t,y,args):

    noise_object, noise_index = args

    noise=noise_object(t)
    if np.isscalar(t):
        perturbation_vector = np.zeros_like(y)
    else:
        perturbation_vector = np.zeros((len(y),len(t)))
    perturbation_vector[noise_index]=noise

    return perturbation_vector



def exp_decay_spectrum(freqs, max_ampl=1.0, cutoff=100):
    """
    Example spectrum function that generates an exponentially decaying spectrum.
    """
    return max_ampl*np.exp(-freqs / cutoff)


def gaussian_spectrum(freqs, max_ampl=1.0, center=0.5, width=0.1):
    """
    Example spectrum function that generates a Gaussian spectrum.
    """
    return max_ampl*np.exp(-0.5 * ((freqs - center) / width) ** 2)

def exp_gaussian_spectrum(freqs, max_ampl=1.0, cutoff=100, center=0.5, width=0.1):
    """
    Example spectrum function that generates a combination of an exponentially decaying spectrum and a Gaussian spectrum.
    """
    return exp_decay_spectrum(freqs, max_ampl=max_ampl, cutoff=cutoff) + gaussian_spectrum(freqs, max_ampl=10*max_ampl, center=center, width=width)

def white_spectrum(freqs, max_ampl=1.0, freq_min=0, freq_max=np.inf):
    """
    Example spectrum function that generates a white noise spectrum (constant power across all frequencies).
    """
    psd= np.zeros_like(freqs)
    psd[ freqs>=freq_min] = max_ampl
    psd[ freqs>freq_max] = 0
    return psd

def realistic_spectrum(freqs, min=0.05,cutoff=6):
    """
    Example spectrum function that generates a more realistic spectrum by combining an exponentially decaying spectrum with a Gaussian peak.
    """
    psd = np.zeros_like(freqs)
    psd [freqs<cutoff]= freqs[freqs<cutoff]**(-5/3) 
    psd [freqs<min] = 0
    #plt.plot(freqs[freqs<cutoff]*2*np.pi, psd[freqs<cutoff])
    #plt.show()
    return psd

def realistic_spectrum_with_peak(freqs, min=0.05,cutoff=6, center=0.5, width=0.1):
    """
    Example spectrum function that generates a more realistic spectrum by combining an exponentially decaying spectrum with a Gaussian peak.
    """
    psd = realistic_spectrum(freqs, min=min, cutoff=cutoff)*0.0001
    psd += gaussian_spectrum(freqs, max_ampl=0.1, center=center, width=width)
    return psd

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

    shift_matrix_obj, model, offset =args

    if offset is None:
        offset=np.zeros_like(model.fixed_point)

    y_dot= shift_matrix_obj.shift_matrix @ (y[:len(model.fixed_point)]-model.fixed_point+offset)

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

    shift_matrix_obj, model, offset =args

    y_dot= shift_matrix_obj.shift_matrix @ np.kron((y[:len(model.fixed_point)]-model.fixed_point+offset),(y[:len(model.fixed_point)]-model.fixed_point+offset))

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
        perturbation_value = np.concatenate((np.zeros(len(y)-len(perturbation_value)),perturbation_value))
        
        y_dot+=perturbation_value

    # adding the shift if provided
    if shift is not None:
        shift_value= shift(t,y,shift_args)
        #accounting for models of higher order than the shift:
        y_dot+=np.concatenate((np.zeros(len(y)-len(shift_value)),shift_value))
    
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
        print("At time {}/{}".format(np.round(times[step],decimals=1),t_final)+" "*10, end="\r")
        y_vals[:,step]=integrator.integrate(integrator.t+dt)
    return times, y_vals


if __name__ == "__main__":
    #print(np.shape(spectrum_noise(exp_decay_spectrum, func_params={'max_ampl': 1.0, 'cutoff': 10}, samples=1024, rate=10)))
    freqs= np.fft.rfftfreq(8000, 1.0/20)
    #plt.plot(freqs, exp_decay_spectrum(freqs, max_ampl=1.0, cutoff=1/(2*np.pi))+gaussian_spectrum(freqs, max_ampl=1.0, center=0.5, width=0.05))
    #plt.plot(np.arange(8000)[:100]*0.1, spectrum_noise(exp_decay_spectrum, func_params={'max_ampl': 1.0, 'cutoff': 3/(2*np.pi)} , samples=8000, rate=20)[1][:100])
    #plt.plot(np.arange(8000)[:100]*0.1, spectrum_noise(gaussian_spectrum, func_params={'max_ampl': 1.0, 'center': 0.5, 'width': 0.05} , samples=8000, rate=20)[1][:100])
    psd,noise=spectrum_noise(exp_gaussian_spectrum, func_params={'max_ampl': 1.0, 'cutoff': 3/(2*np.pi), 'center': 0.5, 'width': 0.05} , samples=8000, rate=20)
    plt.plot(freqs, psd)
    plt.plot(np.arange(8000)[:100]*0.1, noise[:100])
    plt.show()