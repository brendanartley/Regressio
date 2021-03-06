import numpy as np
import matplotlib.pyplot as plt

class linear_regression():
    """Linear regression model. 

    A regression model that models the relationship between a variable x 
    and a variable y using an nth degree polynomial. Degree > 10 is 
    numerically unstable in the OLS calculation. Consider using a natural 
    cubic spline if a degree 10 polynomial underfits.

    Args
    ---------
        degree: the degree of the polynomial

    Attributes
    ---------
        degree (int): the degree of the polynomial
        ws (arr[int]): stored model weights
        range (arr[float]): stored range of the x-values
        rmse (float): stored RMSE from model training

    Raises
    ---------
        TypeError: if degree is not a number
        ValueError: if degree is < 0 or > 10

    Example
    ---------
    >>> from regressio.models import linear_regression
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(100)
    >>> model = linear_regression(degree=5)
    >>> model.fit(x, y, plot=True)
    """

    def __init__(self, degree):
        self.degree = self.check_degree_input(degree)
        self.ws = np.random.random(size=degree).astype(np.float128)
        self.range = [-1,1]
        self.rmse = None

    def fit(self, rawx, y, plot=False):
        '''
        Given input arrays x and y. Fits the model.
        '''
        # Validate input data
        if len(rawx) != len(y):
            raise ValueError('x and y must be of the same length')
        elif len(rawx) <= 1:
            raise ValueError('len(x) must be > 1')

        # Stores min, max of x for plot_polynomial function
        self.range = [rawx.min(), rawx.max()]

        # Calculate ordinary least squares
        rawx = rawx.reshape(-1,1)
        x = np.hstack([(rawx**i) for i in range(self.degree+1)])
        xTx = x.T.dot(x)
        xTx_inv = np.linalg.inv(xTx)
        ws = xTx_inv.dot(x.T.dot(y))
        self.ws = ws # storing model weights
        
        # Calculating MSE
        training_mse = self.MSE(x, y)

        # Store model weights and plot results
        self.ws = ws
        if plot:
            self.plot_model(rawx, y, training_mse)

    def predict(self, x):
        '''
        Given an input array x. Make predictions.
        '''
        values = np.hstack([self.ws[i] * (x ** i) for i in range(0, len(self.ws))])
        preds = np.sum(values, axis=-1)
        return preds

    def MSE(self, x, y):
        '''
        Mean squared error helper function.
        '''
        y_hat = x.dot(self.ws)
        loss = np.mean((y - y_hat) ** 2)
        return loss

    def plot_model(self, x, y, training_mse):
        '''
        Plots the models hypothetical predictions, MSE, and true data points.
        '''
        # Creates model prediction line
        modelx = np.linspace(self.range[0], self.range[1], 1000).reshape([-1, 1])
        expanded = np.hstack([self.ws[i] * (modelx ** i) for i in range(0, len(self.ws))])
        modely = np.sum(expanded, axis=-1)

        # Plot training data
        plt.scatter(x, y)
        
        # Plots model prediction line
        plt.plot(modelx, modely, color='tab:orange')
        plt.title("{}, Degree: {}, MSE: {:.8f}".format(type(self).__name__, self.degree, training_mse))
        plt.show()

    @staticmethod
    def check_degree_input(degree):
        '''
        Validates degree input.
        '''
        if type(degree) != int:
            raise TypeError('Degree must be an integer')
        elif degree < 0 or degree > 10:
            raise ValueError('0 <= degree <= 10. Model is unstable for degree > 10.')
        return degree

class linear_spline():
    """Linear spline model (aka. piecewise simple linear regression).

    A linear interpolation model that fits a simple linear regression model 
    to a variable x and a variable y, in 2 or more segments. The function in 
    each segment is estimated using ordinary least squares. Values that fall 
    outside the range of the training data are predicted as endpoint 
    values.

    Args
    ---------
        knots: number of linear segments + 1

    Attributes
    ---------
        knots (int): the degree of the polynomial
        slopes (arr[float]): stores piecewise linear model slopes
        last_binvals (arr[float]): stores last y-value of each bin
        knot_vals (arr[float]): stores each knot value

    Raises
    ---------
        TypeError: if knots is not a number
        ValueError: if knots < 2

    Example
    ---------
    >>> from regressio.models import linear_spline
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(100)
    >>> model = linear_spline(knots=10)
    >>> model.fit(x, y, plot=True)
    """

    def __init__(self, knots):
        self.knots = self.check_knots_input(knots)
        self.slopes = np.zeros(knots)
        self.last_binvals = np.zeros(knots)
        self.knot_vals = None #set knot values model.fit()
        
    def fit(self, x, y, plot=False):
        '''
        Given input arrays x and y. Fits the model.
        '''
        # check inputs are the same length
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')

        ysum = 0
        last_binval = 0
        self.knot_vals = np.linspace(x.min(), x.max(), num=self.knots)

        for i in range(self.knots-1):
            # Select points in the current bin
            bin_x = x[np.logical_and(x>=self.knot_vals[i], x<=self.knot_vals[i+1])]
            self.check_bin(i, bin_x)

            # Select bin ys and shift bin starting x and starting y to 0,0
            bin_y = y[np.logical_and(x>=self.knot_vals[i], x<=self.knot_vals[i+1]).flatten()]
            bin_y = bin_y - last_binval
            bin_x = bin_x - self.knot_vals[i]
            
            if i == 0:
                # Find best fit line for the bin
                ws = self.OLS_linear_spline(bin_x, bin_y, first=True)

                # First value has no constraint on intercept
                self.last_binvals[i] = ws[0]
                self.slopes[i] = ws[1]

                # Calculating last value in bin
                last_binval = self.line(self.knot_vals[1] - self.knot_vals[0], ws[1], ws[0])
                self.last_binvals[i+1] = last_binval
                
            else:  
                # Find best fit line for the bin
                slope = self.OLS_linear_spline(bin_x, bin_y)

                # Store the last value in the bin (to add as intercept onto next bin)
                last_binval = self.line(self.knot_vals[1] - self.knot_vals[0], slope, last_binval)
                self.last_binvals[i+1] = last_binval
                self.slopes[i] = slope

            ysum = self.last_binvals[i+1]
    
        if plot:
            # plot fit model
            self.plot_model(x, y)

    def predict(self, x):
        '''
        Given a 1-dimenional numpy array, make predictions.
        '''
        preds = np.zeros(len(x))
        # iterating every x in input array
        for i in range(len(x)):
            broke = False
            x_raw = x[i] - self.knot_vals[0]
            # if less than starting bin set to starting value
            if x[i] <= self.knot_vals[0]:
                preds[i] = self.last_binvals[0]
            else:
                # iterating over every bin  
                for j in range(len(self.knot_vals)-1):
                    if x[i] > self.knot_vals[j] and x[i] < self.knot_vals[j+1]:
                        preds[i] = self.line(x_raw, self.slopes[j], self.last_binvals[j])
                        broke = True
                        break
                    else:
                        # shift x until the left of bin at origin
                        x_raw -= self.knot_vals[1] - self.knot_vals[0]
                # if x > last_bin set to last_binval
                if broke == False:
                    preds[i] = self.last_binvals[j+1]
        return preds
        
    def OLS_linear_spline(self, x, y, first=False):
        '''
        Modified OLS. Intercept constrained for all bins except for the first.
        '''
        x = x.reshape(-1,1)
        if first:
            x = np.hstack([(x**i) for i in range(2)])
        xTx = x.T.dot(x)
        xTx_inv = np.linalg.inv(xTx)
        ws = xTx_inv.dot(x.T.dot(y))
        return ws
        
    def plot_model(self, x, y):
        '''
        Plots the models hypothetical predictions, MSE, and true data points.
        '''
        plt.plot(self.knot_vals, self.last_binvals, color='tab:orange')
        plt.scatter(x, y)

        preds = self.predict(x)
        MSE = np.mean((y - preds) ** 2)
        
        plt.title("{}, Knots: {}, MSE: {:.8f}".format(type(self).__name__, self.knots, MSE))

        for i in range(len(self.knot_vals)):
            plt.axvline(x = self.knot_vals[i], alpha=0.2)
        plt.show()

    @staticmethod
    def check_bin(i, bin_x):
        '''
        Checks each bin to validate before OLS calculation.
        '''
        # Calculating slope + intercept in 1st segment so need >2 xs
        if i == 0 and len(bin_x) < 2:
            raise ValueError('Need at least 2 data points in the 1st segment.')
        # Need >1 x in all other bins
        if len(bin_x) < 1:
            raise ValueError('Need at least 1 data point in every segment but the 1st.')
        # Need >0 non-zero values in matrix
        if np.sum(bin_x) == 0:
            raise ValueError('Need at least 1 non 0 value in bin xs')
            
    @staticmethod
    def line(x, slope, intercept):
        '''
        Simple line function.
        '''
        return slope*x + intercept

    @staticmethod
    def check_knots_input(knots):
        '''
        Validates input knot parameter.
        '''
        if type(knots) != int:
            raise TypeError('knots must be an integer')
        if knots < 2:
            raise ValueError('knots must be >= 1')
        return knots

class isotonic_regression(linear_spline):
    """Isotonic regression model (strictly increasing linear spline).

    Child of the linear_spline class. Differs in that the slope
    of each piecewise model must be greater than or equal to 0.

    Args
    ---------
        knots: number of linear segments + 1

    Attributes
    ---------
        knots (int): the degree of the polynomial
        slopes (arr[float]): stores piecewise linear model slopes
        last_binvals (arr[float]): stores last y-value of each bin
        knot_vals (arr[float]): stores each knot value

    Raises
    ---------
        TypeError: if knots is not a number
        ValueError: if knots < 2

    Example
    ---------
    >>> from regressio.models import isotonic_regression
    >>> from regressio.datagen import generate_isotonic_sample
    >>> x, y = generate_isotonic_sample(100)
    >>> model = isotonic_regression(knots=12)
    >>> model.fit(x, y, plot=True)
    """

    def fit(self, x, y, plot=False):
        '''
        Given input arrays x and y. Fits the model.
        '''
        # check inputs are the same length
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')

        ysum = y.min()
        last_binval = y.min()
        self.last_binvals[0] = y.min()
        self.knot_vals = np.linspace(x.min(), x.max(), num=self.knots)

        for i in range(self.knots-1):
            # Select points in the current bin
            bin_x = x[np.logical_and(x>=self.knot_vals[i], x<=self.knot_vals[i+1])]
            self.check_bin(i, bin_x)

            # Select bin ys and shift bin starting x and starting y to 0,0
            bin_y = y[np.logical_and(x>=self.knot_vals[i], x<=self.knot_vals[i+1]).flatten()]
            bin_y = bin_y - last_binval
            bin_x = bin_x - self.knot_vals[i]

            # Find best fit line for the bin
            slope = self.ISO_OLS(bin_x, bin_y)
            self.slopes[i] = slope

            # Store the last value in the bin (to add onto next bin)
            last_binval = self.line(self.knot_vals[1] - self.knot_vals[0], slope, last_binval)
            self.last_binvals[i+1] = last_binval
            ysum += last_binval
    
        if plot:
            # plot fit model
            self.plot_model(x, y)

    @staticmethod
    def ISO_OLS(x, y):
        '''
        Isotonic OLS. 
        Returns slope if positive or 0 if not. 
        '''
        x = x.reshape(-1,1)
        xTx = x.T.dot(x)
        xTx_inv = np.linalg.inv(xTx)
        slope = xTx_inv.dot(x.T.dot(y))
        return max(np.asarray([0]), slope)

class bin_regression():
    """Bin regression model.

    Bin regression is when a data sample is divided into intervals, and the prediction
    for each interval is the mean value of data points in the bin.

    Args
    ---------
        bins: the number of bins

    Attributes
    ---------
        bins (int): the number of bins
        bin_ys (arr[float]): the mean value of each bin
        bin_vals (arr[float]): the endpoints of each bin

    Raises
    ---------
        TypeError: if bins is not a number
        ValueError: if bins < 2

    Example
    ---------
    >>> from regressio.models import bin_regression
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(150)
    >>> model = bin_regression(bins=8)
    >>> model.fit(x, y, plot=True)
    """

    def __init__(self, bins):
        self.bins = self.check_bins_input(bins) # number of bins
        self.bin_ys = np.zeros(bins) # stores each bin y value
        self.bin_vals = None #set bin values model.fit()

    def fit(self, x, y, plot=False):
        '''
        Given input arrays x and y. Fits the model.
        '''
        # check inputs are the same length
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')

        self.bin_vals = np.linspace(x.min(), x.max(), num=self.bins+1)

        for i in range(self.bins):
            # Select bin ys and shift bin starting x and starting y to 0,0
            bin_y = y[np.logical_and(x>=self.bin_vals[i], x<=self.bin_vals[i+1]).flatten()]
            self.check_bin(i, bin_y)
            self.bin_ys[i] = np.mean(bin_y)
    
        if plot:
            # plot fit model
            self.plot_model(x, y)

    def predict(self, x):
        '''
        Given a 1-dimenional numpy array, make predictions.
        '''
        preds = np.zeros(len(x))
        # iterating every x in input array
        for i in range(len(x)):
            broke = False
            if x[i] <= self.bin_vals[0]:
                preds[i] = self.bin_ys[0]
            else:
                # iterating until in correct bin
                for j in range(len(self.bin_vals)-1):
                    if x[i] > self.bin_vals[j] and x[i] < self.bin_vals[j+1]:
                        preds[i] = self.bin_ys[j]
                        broke = True
                        break
                if broke == False:
                    preds[i] = self.bin_ys[-1]
        return preds
        
    def plot_model(self, x, y):
        '''
        Plots the models hypothetical predictions, MSE, and true data points.
        '''
        # Plot each bin value twice (left and right limit).
        plt.plot(np.repeat(self.bin_vals, 2)[1:-1], np.repeat(self.bin_ys, 2), color='tab:orange')
        plt.scatter(x, y)

        preds = self.predict(x)
        MSE = np.mean((y - preds) ** 2)
        
        plt.title("{}, Bins: {}, MSE: {:.8f}".format(type(self).__name__, self.bins, MSE))

        for i in range(len(self.bin_vals)):
            plt.axvline(x = self.bin_vals[i], alpha=0.2)
        plt.show()

    @staticmethod
    def check_bin(i, bin_x):
        '''
        Checks each bin has at least one data point.
        '''
        if len(bin_x) < 1:
            raise ValueError('Need at least 1 data point in every segment but the 1st.')
    
    @staticmethod
    def check_bins_input(bins):
        '''
        Validates input bin parameter.
        '''
        if type(bins) != int:
            raise TypeError('bin must be an integer')
        if bins < 2:
            raise ValueError('bins must be >= 1')
        return bins

class cubic_spline():
    """Cubic spline model.

    Cubic spline is a spline constructed of multiple cubic piecewise 
    polynomials. Where two polynomials meet, the 1st and 2nd derivatives are equal. 
    This makes for a smooth fitting line. Cubic spline is better than high degree 
    polynomials as it oscillates less at its endpoints and between values 
    (ie. mitigates Runge's phenomenon).

    Args
    ---------
        pieces: the number of segments

    Attributes
    ---------
        pieces (int): the degree of the polynomial
        knot_xvals (arr[float]): knot x values
        knot_yvals (arr[float]): knot y values
        ws (arr[float]): piecewise model weights

    Raises
    ---------
        TypeError: if pieces is not an integer
        ValueError: if pieces < 2

    Example
    ---------
    >>> from regressio.models import cubic_spline
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(150)
    >>> model = cubic_spline(pieces=15)
    >>> model.fit(x, y, plot=True)

    Reference
    ----------
    Kong, Qingkai, et al. Python Programming and Numerical Methods: A Guide for 
    Engineers and Scientists. Academic Press, an Imprint of Elsevier, 
    pythonnumericalmethods.berkeley.edu, Accessed 2022. 
    """

    def __init__(self, pieces):
        self.pieces = self.check_input_pieces(pieces)
        self.knot_xvals = None
        self.knot_yvals = None
        self.ws = None # stores weights for each piecewise polynomial

    def fit(self, x, y, plot=False):
        '''
        Given input arrays x and y. Fits the model.
        '''
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')
        if len(x) <= self.pieces:
            raise ValueError('len(x) must be greater than number of pieces')

        # Values must be sorted for knot calculation
        y = y[x.argsort()]
        x = np.sort(x)

        # Find (x,y) for each knot value
        self.knot_xvals = np.linspace(x.min(), x.max(), num=self.pieces+1)
        self.knot_yvals = self.find_knots(self.knot_xvals, x, y)
        
        # Calculate weights of each piecewise function
        self.ws = self.calc_piecewise_weights(self.knot_xvals, self.knot_yvals)

        # Plot model
        if plot:
            self.plot_model(x, y)

    def calc_piecewise_weights(self, x, y):
        '''
        Given x, y, returns the weights for the piecewise polynomials. Constructs 
        system of equations in matrix form, and solves.
        '''
        # Create matrix and vector of system of equations
        A = np.zeros((4*(self.pieces), 4*(self.pieces)))
        b = np.concatenate([y[:1], np.repeat(y[1:-1], 2), y[-1:], np.zeros(2*(self.pieces))])

        # Add function constraints
        # Each knot must be on both intersecting functions (with the exception of the endpoints)
        xi = 0
        for i in range(self.pieces):
            row = i*2
            col = i*4

            A[row][col:col+4] = [x[xi]**3, x[xi]**2, x[xi], 1]
            xi += 1
            A[row+1][col:col+4] = [x[xi]**3, x[xi]**2, x[xi], 1]

        # Add derivative constraints
        # (1st + 2nd derivatives at intersecting knots must be equal)
        col = 0
        xi = 0
        for i in range(self.pieces-1):
            col = i*4
            row += 2
            xi += 1

            # 1st derivative
            A[row][col:col+3] = [3*(x[xi]**2), 2*x[xi], 1]
            A[row][col+4:col+7] = [-(3*(x[xi]**2)), -(2*x[xi]), -1]

            # 2nd derivative
            A[row+1][col:col+2] = [6*(x[xi]), 2]
            A[row+1][col+4:col+6] = [-(6*(x[xi])), -2]

        # Add endpoint contraints (2nd derivatives at endpoints equal zero.)
        A[-2][:2] = [6*x[0], 2]
        A[-1][-4:-2] = [6*x[-1], 2]

        # Solve the system of equations
        ws = np.dot(np.linalg.inv(A), b)
        return ws

    def predict(self, xs):
        '''
        Given a set of x values, makes predictions.
        '''
        # Array for predictions (assigning is faster than append)
        ys = np.zeros(len(xs), dtype=np.float64) # use float for precision

        ws_pos = 0
        insert_pos = 0
        for i in range(self.pieces+1):
            # Add points before startpoint
            if i == 0:
                ys_to_add = self.polynomial(xs[xs<=self.knot_xvals[i]], self.ws[ws_pos:ws_pos+4][::-1])
                ys[:len(ys_to_add)] = ys_to_add
            # all other pieces
            else:
                ys_to_add = self.polynomial(xs[np.logical_and(xs>self.knot_xvals[i-1], xs<=self.knot_xvals[i])], self.ws[ws_pos:ws_pos+4][::-1])
                ws_pos+=4

            ys[insert_pos:insert_pos + len(ys_to_add)] = ys_to_add
            insert_pos += len(ys_to_add)

        # Add points after endpoint
        ws_pos-=4
        ys_to_add = self.polynomial(xs[xs>self.knot_xvals[i]], self.ws[ws_pos:ws_pos+4][::-1])
        ys[insert_pos:insert_pos + len(ys_to_add)] = ys_to_add
        return ys

    def plot_model(self, x, y):
        '''
        Plots the models hypothetical predictions, MSE, and true data points.
        '''
        # Plot hypothetical model values
        modelx = np.linspace(self.knot_xvals[0], self.knot_xvals[-1], self.pieces*10)
        modely = self.predict(modelx)
        plt.plot(modelx, modely, color='tab:orange')

        # Plot data values
        plt.scatter(x, y)
        
        MSE = np.mean((y - self.predict(x)) ** 2)
        plt.title("{}, Pieces: {}, MSE: {:.8f}".format(type(self).__name__, self.pieces, MSE))

        # Add piece boundaries to plot
        for i in range(len(self.knot_xvals)):
            plt.axvline(x = self.knot_xvals[i], alpha=0.2)
        plt.show()

    @staticmethod
    def find_knots(knot_vals, xs, ys):
        '''
        Helper function to find the y-value for each knot in xs.
        '''
        knot_ys = np.zeros(len(knot_vals))
        for i, knot in enumerate(knot_vals):
            index = np.searchsorted(xs, int(knot)) #binary search as xs are sorted
            # Use data point on knot
            if int(knot) == knot:
                knot_ys[i] = ys[index]
            # Used mean between closest left + right data points
            else:
                knot_ys[i] = np.mean(ys[index:index+2])
        return knot_ys

    @staticmethod
    def polynomial(values, weights):
        '''
        Given a set of values and weights, returns the values after being passed through
        a polynomial with the given weights.
        '''
        if len(values) == 0:
            return np.empty(0)
        values = values.reshape(-1, 1)
        # weights are assumed to be in order 0, 1, ..., n-1
        expanded = np.hstack([weights[i] * (values ** i) for i in range(0, len(weights))])
        return np.sum(expanded, axis=1)

    @staticmethod
    def check_input_pieces(pieces):
        '''
        Validates input pieces.
        '''
        if type(pieces) != int:
            raise TypeError('pieces must be an integer')
        if pieces < 2:
            raise ValueError('pieces must be >= 2')
        return pieces

class natural_cubic_spline(cubic_spline):
    """Natural cubic spline model.

    Child of the cubic_spline class. Differs in that the spline extrapolates 
    linearly beyond its knot boundaries.

    Args
    ---------
        pieces: the number of segments

    Attributes
    ---------
        pieces (int): the degree of the polynomial
        knot_xvals (arr[float]): knot x values
        knot_yvals (arr[float]): knot y values
        ws (arr[float]): piecewise model weights

    Raises
    ---------
        TypeError: if pieces is not an integer
        ValueError: if pieces < 2

    Example
    ---------
    >>> from regressio.models import natural_cubic_spline
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(200)
    >>> model = natural_cubic_spline(pieces=10)
    >>> model.fit(x, y, plot=True)

    Reference
    ----------
    Kong, Qingkai, et al. Python Programming and Numerical Methods: A Guide for 
    Engineers and Scientists. Academic Press, an Imprint of Elsevier, 
    pythonnumericalmethods.berkeley.edu, Accessed 2022. 
    """
    def __init__(self, pieces):
        self.pieces = self.check_input_pieces(pieces)
        self.knot_xvals = None
        self.knot_yvals = None
        self.ws = None # stores weights for each piecewise polynomial

    def predict(self, xs):
        '''
        Given a set of x values, makes predictions.
        '''
        # Array for predictions (assigning is faster than append)
        ys = np.zeros(len(xs), dtype=np.float64) # use float for precision

        ws_pos = 0
        insert_pos = 0
        for i in range(self.pieces+1):
            # Add points before startpoint
            if i == 0:
                # Naive calculation of slope at endpoint
                p1 = self.polynomial(self.knot_xvals[i:i+1], self.ws[ws_pos:ws_pos+4][::-1])
                p2 = self.polynomial(self.knot_xvals[i:i+1] + 0.01, self.ws[ws_pos:ws_pos+4][::-1])
                slope = (p2 - p1) / 0.01
                # Calculating values with a shift based on starting knot xvalue
                ys_to_add = self.line(xs[xs<=self.knot_xvals[i]] - self.knot_xvals[0], slope, p2)
                ys[:len(ys_to_add)] = ys_to_add

            # all other pieces
            else:
                ys_to_add = self.polynomial(xs[np.logical_and(xs>self.knot_xvals[i-1], xs<=self.knot_xvals[i])], self.ws[ws_pos:ws_pos+4][::-1])
                ws_pos+=4
            
            # inserting predictions to array, moving pointer
            ys[insert_pos:insert_pos + len(ys_to_add)] = ys_to_add
            insert_pos += len(ys_to_add)

        # Add points after endpoint
        ws_pos-=4
        
        # Naive calculation of slope
        p1 = self.polynomial(self.knot_xvals[-1:], self.ws[ws_pos:ws_pos+4][::-1])
        p2 = self.polynomial(self.knot_xvals[-1:] + 0.01, self.ws[ws_pos:ws_pos+4][::-1])
        slope = (p2 - p1) / 0.01
        ys_to_add = self.line(xs[xs>self.knot_xvals[i]] - self.knot_xvals[-1], slope, p2)
        
        # inserting points past the endpoint to predictions array
        ys[insert_pos:insert_pos + len(ys_to_add)] = ys_to_add
        return ys
    
    @staticmethod
    def line(x, slope, intercept):
        return (slope*x) + intercept

class exponential_smoother():
    """Exponential smoothing model.

    An iterative model that make predictions based on the weighted moving average of past 
    predictions with exponentially decreasing weight.

    This model uses mean initialization for the first forecast value. This has no 
    significant effect compared to MSE optimization when len(y) >= 10. See more on initial
    forecasting values in the reference below.

    Args
    ---------
        alpha: the starting weight of previous value

    Attributes
    ---------
        alpha: the starting weight of previous value

    Raises
    ---------
        TypeError: if alpha is not a floating point
        ValueError: if alpha is not >0 and <1

    Example
    ---------
    >>> from regressio.models import exponential_smoother
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(200)
    >>> model = exponential_smoother(alpha=0.1)
    >>> model.fit(x, y, plot=True)

    Reference
    ----------
    Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 
    3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed July 2022.
    """
    
    def __init__(self, alpha=0.2):
        self.alpha = self.check_alpha(alpha)

    def fit(self, x, y, plot=False):
        '''
        Given arrays x and y, returns exponentially smoothed y values.
        '''
        if len(y) <= 2:
            raise ValueError('len(y) must be greater than 2')
        
        # Array for storing smoothed values
        smoothed_ys = np.zeros_like(y)
        smoothed_ys[0] = np.mean(y[:10]) # Initializing initial value using Mean

        # Iterating y values
        for i in range(1, len(y)):
            smoothed_ys[i] = self.alpha*y[i-1] + (1-self.alpha)*smoothed_ys[i-1]

        if plot:
            self.plot_model(x, y, smoothed_ys)

    def plot_model(self, x, y, smoothed_ys):
        '''
        Plots the smoothed model, MSE, and true data points.
        '''
        # Plot smoothed model values
        plt.plot(x, smoothed_ys, color='tab:orange')
        
        # Plot actual data values
        plt.scatter(x, y)
        
        # MSE and Title
        MSE = np.mean((y - smoothed_ys) ** 2)
        plt.title("{}, MSE: {:.8f}".format(type(self).__name__, MSE))
        plt.show()

    @staticmethod
    def check_alpha(alpha):
        '''
        Validates alpha input.
        '''
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0 or alpha > 1:
            raise ValueError('alpha must be between 0 and 1')
        return alpha

if __name__ == '__main__':
    main()