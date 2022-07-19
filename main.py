import numpy as np
import matplotlib.pyplot as plt

from regressio.models import *
from regressio.datagen import generate_isotonic_sample, generate_random_walk

plt.rcParams['figure.figsize'] = (10, 5)

def main():
    np.random.seed(1)
    x, y = generate_random_walk(100)

    model = linear_regression(degree=5)
    model.fit(x, y, plot=True)
    return
    
if __name__ == '__main__':
    main()