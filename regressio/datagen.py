import numpy as np
import matplotlib.pyplot as plt

def generate_random_walk(n, noise=1, plot=False):
    """Given a number n, returns a random walk sample. 

    Args
    ---------
        n: the number of data points in the sample
        noise: random walk step size
        plot: boolean to plot result or not

    Returns
    ---------
        x: x values of the data sample
        y: y values of the data sample

    Example
    ---------
    >>> from regressio.models import linear_regression
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(100)
    """
    if n < 1:
        raise ValueError('n must be >= 1')
    x = np.arange(n, dtype=np.float64)
    y = np.cumsum(np.concatenate([[0], np.random.normal(0, noise, n-1)]))
    if plot:
        plt.scatter(x, y)
        plt.show()
    return x, y

def generate_isotonic_sample(n, noise=1, plot=False):
    """Given a number n, generates an increasing sample.
    For a strictly increasing sample set noise to 0.
 
    Args
    ---------
        n: the number of data points in the sample
        noise: random walk step size
        plot: boolean to plot result or not

    Returns
    ---------
        x: x values of the data sample
        y: y values of the data sample

    Example
    ---------
    >>> from regressio.models import linear_regression
    >>> from regressio.datagen import generate_isotonic_sample
    >>> x, y = generate_isotonic_sample(100)
    """
    def gen(x):
        # Creates periods of varying rates of increase
        if (x%10)%2 == 0:
            return np.random.choice([0, 0.25, 0.5], p=[0.8,0.15,0.05])
        else:
            return np.random.choice([0, 0.1], p=[0.9,0.1])

    x = np.arange(n, dtype=np.float64)
    y = np.asarray([gen(val) for val in x]) + np.random.normal(0, noise/10, size=n)
    y = np.cumsum(np.concatenate([[0], y[:-1]]))
    if plot:
        plt.scatter(x, y)
        plt.show()
    return x, y

def main():
    return
    
if __name__ == '__main__':
    main()