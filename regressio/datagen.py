import numpy as np
import matplotlib.pyplot as plt

def generate_random_walk(n, noise=1, plot=False):
    '''
    Given a number of data points n, 
    returns a numpy array of a random walk with length n. 
    '''
    if n < 1:
        raise ValueError('n must be >= 1')
    x = np.arange(n, dtype=np.float64).reshape([-1,1])
    y = np.cumsum(np.concatenate([[0], np.random.normal(0, noise, n-1)]))
    if plot:
        plt.figure(figsize=(10, 5))
        plt.scatter(x, y, s=5)
        plt.show()
    return x, y

def main():
    return
    
if __name__ == '__main__':
    main()