import numpy as np
import matplotlib.pyplot as plt

from regressio.models import *
from regressio.datagen import generate_isotonic_sample, generate_random_walk

plt.rcParams['figure.figsize'] = (10, 5)

def main():
    np.random.seed(0)
    x, y = generate_random_walk(150)

    model = cubic_spline(pieces=15)
    model.fit(x, y, plot=True)

    np.random.seed(1)
    x, y = generate_random_walk(100)

    model = linear_regression(degree=5)
    model.fit(x, y, plot=True)

    # np.random.seed(2)
    # x, y = generate_random_walk(100)

    # model = linear_interpolation(knots=10)
    # model.fit(x, y, plot=True)

    np.random.seed(4)
    x, y = generate_isotonic_sample(100)

    model = isotonic_regression(knots=12)
    model.fit(x, y, plot=True)

    # np.random.seed(5)
    # x, y = generate_random_walk(150)

    # model = bin_regression(bins=8)
    # model.fit(x, y, plot=True)
    return
    
if __name__ == '__main__':
    main()