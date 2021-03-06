<h1 align="center">
<img src="./imgs/logo.svg" width="300">
</h1><br>

Regressio is a python module for univariate regression, interpolation, and smoothing.

The available models are:
- Linear Regression
- Linear Spline
- Isotonic Regression
- Bin Regression
- Cubic Spline
- Natural Cubic Spline
- Exponential Smoothing

There are also functions implemented to generate data samples.

The available data generators are:
- Random Walk
- Isotonic Sample

## Installation

Regressio is supported in Python 3.6+ or above and requires only NumPy and Matplotlib.

`pip install git+https://github.com/brendanartley/Regressio`

## Example Usage

Cubic spline. 

```python
# Import modules + classes
from regressio.models import cubic_spline
from regressio.datagen import generate_random_walk
import numpy as np
import matplotlib.pyplot as plt

# Set figsize and seed
plt.rcParams['figure.figsize'] = (10, 5)
np.random.seed(0)

# Generate data sample
x, y = generate_random_walk(150)

# Fit model and plot result
model = cubic_spline(pieces=15)
model.fit(x, y, plot=True)
```
<img alt="Cubic Spline" src="imgs/cubic_spline.png" width="550">

Linear regression.

```python
# Import modules + classes
from regressio.models import linear_regression
from regressio.datagen import generate_random_walk
import numpy as np
import matplotlib.pyplot as plt

# Set figsize and seed
plt.rcParams['figure.figsize'] = (10, 5)
np.random.seed(1)

# Generate data sample
x, y = generate_random_walk(100)

# Fit model and plot result
model = linear_regression(degree=5)
model.fit(x, y, plot=True)
```
<img alt="Linear Regression" src="imgs/linear_regression.png" width="550">

Isotonic regression.

```python
# Import modules + classes
from regressio.models import isotonic_regression
from regressio.datagen import generate_isotonic_sample
import numpy as np
import matplotlib.pyplot as plt

# Set figsize and seed
plt.rcParams['figure.figsize'] = (10, 5)
np.random.seed(4)

# Generate data sample
x, y = generate_isotonic_sample(100)

# Fit model and plot result
model = isotonic_regression(knots=12)
model.fit(x, y, plot=True)
```
<img alt="Isotonic Regression" src="imgs/isotonic_regression.png" width="550">

For more examples, navigate to the [examples.ipynb](examples.ipynb) file in this repository.

## References

- Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed July 2022. 

- Kong, Qingkai, et al. Python Programming and Numerical Methods: A Guide for Engineers and Scientists. Academic Press, an Imprint of Elsevier, pythonnumericalmethods.berkeley.edu, Accessed 2022. 
