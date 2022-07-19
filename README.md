<h1 align="center">
<img src="./imgs/logo.svg" width="300">
</h1><br>

A simple python library of regression models.

## Example Usage

Linear regression.

```
import numpy as np
import matplotlib.pyplot as plt
from regressio.models import linear_regression
from regressio.datagen import generate_random_walk

np.random.seed(1)
plt.rcParams['figure.figsize'] = (10, 5)

x, y = generate_random_walk(n=100)

model = linear_regression(degree=5)
model.fit(x, y, plot=True)
```
<img alt="Linear Regression" src="imgs/linear_regression.png" width="550">

Linear Interpolation (aka. Piecewise Linear Regression).

```
import numpy as np
import matplotlib.pyplot as plt
from regressio.models import linear_interpolation
from regressio.datagen import generate_random_walk

np.random.seed(2)
plt.rcParams['figure.figsize'] = (10, 5)

x, y = generate_random_walk(n=100)

model = linear_interpolation(knots=10)
model.fit(x, y, plot=True)
```
<img alt="Linear Interpolation" src="imgs/linear_interpolation.png" width="550">

Isotonic regression. Strictly increasing linear interpolation.

```
import numpy as np
import matplotlib.pyplot as plt
from regressio.models import isotonic_regression
from regressio.datagen import generate_isotonic_sample


np.random.seed(4)
plt.rcParams['figure.figsize'] = (10, 5)

x, y = generate_isotonic_sample(n=100)

model = isotonic_regression(knots=12)
model.fit(x, y, plot=True)
```
<img alt="Isotonic Regression" src="imgs/isotonic_regression.png" width="550">

Bin regression.

```
import numpy as np
import matplotlib.pyplot as plt
from regressio.models import bin_regression
from regressio.datagen import generate_random_walk

np.random.seed(5)
plt.rcParams['figure.figsize'] = (10, 5)

x, y = generate_random_walk(n=100)

model = isotonic_regression(knots=8)
model.fit(x, y, plot=True)
```
<img alt="Bin Regression" src="imgs/bin_regression.png" width="550">

More examples to come in the [notebooks folder](notebooks/).
