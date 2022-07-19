import numpy as np
import matplotlib.pyplot as plt

from regressio.models import linear_regression, linear_interpolation, isotonic_regression
from regressio.datagen import generate_isotonic_sample, generate_random_walk

def main():
    np.random.seed(1)
    plt.rcParams['figure.figsize'] = (10, 5)

    x, y = generate_random_walk(n=100)

    model = linear_interpolation(knots=10)
    model.fit(x, y, plot=True)
    return
    
if __name__ == '__main__':
    main()