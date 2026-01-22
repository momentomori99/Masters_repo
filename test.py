import numpy as np
import matplotlib.pyplot as plt

def generate_poisson_array(n, lambda_param):
    """
    Generate an array of length n with values following a Poisson distribution.
    
    Parameters:
    -----------
    n : int
        Length of the array to generate
    lambda_param : int
        The lambda parameter (mean/mode) of the Poisson distribution.
        This will be the most probable value in the distribution.
    
    Returns:
    --------
    numpy.ndarray
        Array of length n with Poisson-distributed values
    """
    return np.random.poisson(lam=lambda_param, size=n)


# Example usage
if __name__ == "__main__":
    # Generate array with lambda = 5
    sample_array = generate_poisson_array(n=100000, lambda_param=150)
    print(len(sample_array))
    for i in range(len(sample_array)):
        print(sample_array[i])
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(sample_array, bins=1000, 
             density=True, alpha=0.7, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Poisson Distribution (λ = 5, n = 10000)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Mean: {np.mean(sample_array):.2f}")
    print(f"Most common value: {np.bincount(sample_array).argmax()}")