import numpy as np
import matplotlib.pyplot as plt

from regressio.models import isotonic_regression
from regressio.datagen import generate_isotonic_sample

from regressio.models import linear_regression
from regressio.datagen import generate_random_walk

np.random.seed(1)
plt.rcParams['figure.figsize'] = (10, 5)

def main():
    x, y = generate_random_walk(100)

    model = linear_regression(degree=10)
    model.fit(x, y, plot=True)
    return
    
if __name__ == '__main__':
    main()