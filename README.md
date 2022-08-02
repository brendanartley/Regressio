<h1 align="center">
<img src="https://github.com/brendanartley/Regressio/blob/main/imgs/logo.svg?raw=true" width="300">
</h1>

<p align="center">
    <a href="https://pypi.org/project/regressio/" alt="PYPI">
        <img src="https://img.shields.io/pypi/v/regressio"/>
    </a>
    <a href="https://github.com/brendanartley/Regressio/blob/main/License" alt="MIT License">
        <img src="https://img.shields.io/github/license/brendanartley/regressio"/>
    </a>
    <a href="https://github.com/brendanartley/Regressio/issues" alt="Issues">
        <img src="https://img.shields.io/github/issues/brendanartley/regressio"/>
    </a>
    <a href="https://github.com/brendanartley/Regressio" alt="Stars">
        <img src="https://img.shields.io/github/stars/brendanartley/regressio?style=social"/>
    </a>
</p>

Regressio is a python module for univariate regression, interpolation, and smoothing.

The available models are:
- Linear regression
- Ridge regression
- Linear spline
- Isotonic regression
- Bin regression
- Cubic spline
- Natural cubic spline
- Exponential moving average
- Kernel functions (Gaussian, KNN, Weighted average)

There are also functions implemented to generate data samples.

The available data generators are:
- Random walk
- Isotonic sample

## Installation

Regressio is supported in Python 3.8+ and requires only NumPy and Matplotlib.

```python
pip install regressio --upgrade
```
or
```python
pip install git+https://github.com/brendanartley/Regressio
```

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
model.fit(x, y, plot=True, confidence_interval=0.99)
```
<img alt="Cubic spline" src="https://github.com/brendanartley/Regressio/blob/main/imgs/cubic_spline.png?raw=true" width="650">

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
model.fit(x, y, plot=True, confidence_interval=0.95)
```
<img alt="Linear regression" src="https://github.com/brendanartley/Regressio/blob/main/imgs/linear_regression.png?raw=true" width="650">

Exponential moving average.

```python
# Import modules + classes
from regressio.models import exp_moving_average
from regressio.datagen import generate_isotonic_sample
import numpy as np
import matplotlib.pyplot as plt

# Set figsize and seed
plt.rcParams['figure.figsize'] = (10, 5)
np.random.seed(6)

# Generate data sample
x, y = generate_isotonic_sample(100)

# Fit model and plot result
model = exp_moving_average(alpha=0.2)
model.fit(x, y, plot=True, confidence_interval=0.90)
```
<img alt="Exponential moving average" src="https://github.com/brendanartley/Regressio/blob/main/imgs/exponential_smoother.png?raw=true" width="650">

For more examples, navigate to the [examples.ipynb](https://github.com/brendanartley/Regressio/blob/main/examples.ipynb) file in this repository.

## Contributions

We welcome all to contribute their expertise to the Regressio library. If you are new to open source contributions, [this guide](https://opensource.guide/how-to-contribute/) gives some great tips on how to get started.

If you have a complex feature in mind or find a large bug in the code, please create a detailed issue and we will get to work on it.

## References

- Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed July 2022. 

- Kong, Qingkai, et al. Python Programming and Numerical Methods: A Guide for Engineers and Scientists. Academic Press, an Imprint of Elsevier, pythonnumericalmethods.berkeley.edu, Accessed July 2022. 

- Li, Bao, (2022). Stat 508: Applied Data Mining, Statistical Learning: Stat Online. PennState: Statistics Online Courses, online.stat.psu.edu/stat508, Accessed July 2022.

- Brett, M. (2014, October 26). An introduction to smoothing. Tutorials on imaging, computing and mathematics. matthew-brett.github.io/teaching, Accessed July 2022.