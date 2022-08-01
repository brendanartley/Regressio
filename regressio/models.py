import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist

class smoother(object):
    """
    Abstract class for all smoothing models. This class can not be instantiated 
    directly and is strictly a parent class.  

    Attributes
    ---------
        residuals (arr[int | float]): stored residuals
        smoothed_ys (arr[int | float]): stored smoothed values
        zscore (dict): dictionary of confidence z-scores

    Raises
    ---------
        Exception: if smoother is instantiated directly
    """
    
    def __init__(self):
        self.check_instantiation_type()
        self.residuals = None 
        self.smoothed_ys = None
    
    def get_confidence_interval(self, confidence=0.95):
        """
        Function that bootstraps residuals for standard deviation, and returns a
        band size.
        """
        self.check_ci_input(confidence) # checking valid confidence interval

        # Performs up to 500 bootstraps.
        if len(self.residuals) < 500:
            bs = np.random.choice(self.residuals, (len(self.residuals), len(self.residuals)), replace=True)
        else:
            bs = np.random.choice(self.residuals, (500, len(self.residuals)), replace=True)

        # Mean of the bootstrapped standard deviations
        bs_std = np.mean(np.std(bs, axis=1))

        # Calc Z-score + band_size. Inverse CDF calculation.
        z_score = NormalDist().inv_cdf((1 + confidence) / 2.)
        band_size = z_score * bs_std

        return band_size

    def check_instantiation_type(self):
        """
        Function to enforce abstract class type.
        """
        if type(self) is smoother:
            raise Exception('smoothing_model is an abstract class and cannot be instantiated directly')

    def check_ci_input(self, ci):
        '''
        Validates confidence interval input.
        '''
        if type(ci) != float:
            raise TypeError('Confidence interval must be an float')
        if ci <= 0 or ci > 1:
            raise ValueError('Confidence interval must be greater than 0 and less than 1')
        if ci == 1: #edge case where random number generated is 1
            return 1 - np.finfo(np.float64).eps
        return ci

class linear_regression(smoother):
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
        + attributes from smoother class

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
    >>> model.fit(x, y, plot=True, confidence_interval=0.95)
    """

    def __init__(self, degree):
        smoother.__init__(self)
        self.degree = self.check_degree_input(degree)
        self.ws = np.random.random(size=degree).astype(np.float128)
        self.range = [-1,1]
        self.rmse = None

    def fit(self, x, y, plot=False, confidence_interval=False):
        '''
        Given input arrays x and y. Fits the model.
        '''
        # Validate x and y input
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')
        elif len(x) <= 1:
            raise ValueError('len(x) must be > 1')

        # Store min, max of x for plot_polynomial function
        self.range = [x.min(), x.max()]

        # Calculate ordinary least squares
        x = x.reshape(-1, 1)
        MSE = self.OLS(x, y)

        # Store residuals and smoothed values
        self.smoothed_ys = self.predict(x)
        self.residuals = y - self.smoothed_ys

        # Plot model
        if plot:
            self.plot_model(x, y, MSE, confidence_interval)

    def predict(self, x):
        '''
        Given an input array x. Make predictions.
        '''
        values = np.hstack([self.ws[i] * (x ** i) for i in range(0, len(self.ws))])
        preds = np.sum(values, axis=-1)
        return preds

    def OLS(self, x, y):
        # Construct features 1, x^1, x^2 .. x^n
        x = x.reshape(-1, 1)
        x_features = np.hstack([(x**i) for i in range(self.degree+1)])

        # OLS Calculation
        xTx = x_features.T.dot(x_features)
        xTx_inv = np.linalg.inv(xTx)
        ws = xTx_inv.dot(x_features.T.dot(y))

        # Store weights + return MSE
        self.ws = ws
        MSE = self.MSE(x_features, y)
        return MSE

    def MSE(self, x, y):
        '''
        Mean squared error helper function.
        '''
        y_hat = x.dot(self.ws)
        loss = np.mean((y - y_hat) ** 2)
        return loss

    def plot_model(self, x, y, MSE, confidence_interval=False):
        '''
        Plot the models hypothetical predictions, MSE, and true data points.
        '''
        # Plot hypothetical model values
        modelx = np.linspace(self.range[0], self.range[1], 1000).reshape([-1, 1])
        expanded = np.hstack([self.ws[i] * (modelx ** i) for i in range(0, len(self.ws))])
        modely = np.sum(expanded, axis=-1)
        plt.plot(modelx, modely, color='tab:orange')

        # Plot data
        plt.scatter(x, y)

        # Title + MSE
        plt.title("{}, Degree: {}, MSE: {:.4f}".format(type(self).__name__, self.degree, MSE))

        # Optional: Plot confidence interval
        if confidence_interval:
            band_size = self.get_confidence_interval(confidence_interval)
            plt.fill_between(modelx.flatten(), modely - band_size, modely + band_size, color='tab:blue', alpha=0.2)
            plt.title("{}, MSE: {:.4f}, Confidence Interval: {:.8g}%".format(type(self).__name__, MSE, confidence_interval*100))
        
        plt.show()

    @staticmethod
    def check_degree_input(degree):
        '''
        Validate degree input.
        '''
        if type(degree) != int:
            raise TypeError('Degree must be an integer')
        elif degree < 0 or degree > 10:
            raise ValueError('0 <= degree <= 10. Model is unstable for degree > 10.')
        return degree

class ridge_regression(linear_regression):
    """Ridge regression model.

    Child of the linear_regression class. Differs in that a penalty
    term is added. This penalty term penalizes large squared model 
    weights and reduces the likelihood of overfitting.

    Args
    ---------
        alpha: the magnitude of the penalty term
        degree: the degree of the polynomial

    Attributes
    ---------
        alpha: the magnitude of the penalty term
        + attributes from linear_regression class

    Raises
    ---------
        TypeError: if knots is not a number
        ValueError: if alpha <= 0

    Example
    ---------
    >>> from regressio.models import ridge_regression
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(100)
    >>> model = ridge_regression(degree=5, alpha=0.5)
    >>> model.fit(x, y, plot=True, confidence_interval=0.95)
    
    Reference
    ----------
    Li, Bao, (2022). Stat 508: Applied Data Mining, Statistical 
    Learning: Stat Online. PennState: Statistics Online Courses, 
    online.stat.psu.edu/stat508, Accessed July 2022.
    """

    def __init__(self, degree, alpha=0.1):
        linear_regression.__init__(self, degree)
        self.alpha = self.check_alpha_input(alpha)

    def OLS(self, x, y):
        # Construct features 1, x^1, x^2 .. x^n
        x = x.reshape(-1, 1)
        x_features = np.hstack([(x**i) for i in range(self.degree+1)])
        
        # Construct L2 penalty matrix (Ridge)
        A = np.identity(self.degree+1) * self.alpha
        A[0,0] = 0

        # OLS Calculation
        xTx = x_features.T.dot(x_features)
        xTx_inv = np.linalg.inv(xTx + A)
        ws = xTx_inv.dot(x_features.T.dot(y))

        # Store weights + return MSE
        self.ws = ws
        MSE = self.MSE(x_features, y)
        return MSE

    @staticmethod
    def check_alpha_input(alpha):
        '''
        Validate alpha input.
        '''
        if type(alpha) not in [float, int]:
            raise TypeError('alpha must be an float or int')
        elif alpha <= 0:
            raise ValueError('alpha must be > 0')
        return alpha

class linear_spline(smoother):
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
        + attributes from smoother class

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
    >>> model.fit(x, y, plot=True, confidence_interval=0.90)
    """

    def __init__(self, knots):
        smoother.__init__(self)
        self.knots = self.check_knots_input(knots)
        self.slopes = np.zeros(knots)
        self.last_binvals = np.zeros(knots)
        self.knot_vals = None #set knot values model.fit()
        
    def fit(self, x, y, plot=False, confidence_interval=False):
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

                # Calculate last value in bin
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

        # Store residuals and smoothed values
        self.smoothed_ys = self.predict(x)
        self.residuals = y - self.smoothed_ys

        # Plot model
        if plot:
            self.plot_model(x, y, confidence_interval)

    def predict(self, x):
        '''
        Given a 1-dimenional numpy array, make predictions.
        '''
        preds = np.zeros(len(x))
        # iterate over every x in input array
        for i in range(len(x)):
            broke = False
            x_raw = x[i] - self.knot_vals[0]
            # if less than starting bin set to starting value
            if x[i] <= self.knot_vals[0]:
                preds[i] = self.last_binvals[0]
            else:
                # iterate over every bin  
                for j in range(len(self.knot_vals)-1):
                    if x[i] >= self.knot_vals[j] and x[i] < self.knot_vals[j+1]:
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
        Modified OLS. Intercept is constrained for all bins except for the first.
        '''
        # Create Features
        x = x.reshape(-1,1)
        if first:
            x = np.hstack([(x**i) for i in range(2)]) # include slope if first line
        
        # OLS Calculation
        xTx = x.T.dot(x)
        xTx_inv = np.linalg.inv(xTx)
        ws = xTx_inv.dot(x.T.dot(y))
        return ws
        
    def plot_model(self, x, y, confidence_interval=False):
        '''
        Plots the models hypothetical predictions, MSE, and true data points.
        '''
        # Plot model + data points
        plt.plot(self.knot_vals, self.last_binvals, color='tab:orange')
        plt.scatter(x, y)

        # MSE + title
        MSE = np.mean((self.residuals) ** 2)
        plt.title("{}, Knots: {}, MSE: {:.8f}".format(type(self).__name__, self.knots, MSE))

        # Optional: Plot confidence interval
        if confidence_interval:
            band_size = self.get_confidence_interval(confidence_interval)
            plt.fill_between(x.flatten(), self.smoothed_ys - band_size, self.smoothed_ys + band_size, color='tab:blue', alpha=0.2)
            plt.title("{}, MSE: {:.4f}, Confidence Interval: {:.8g}%".format(type(self).__name__, MSE, confidence_interval*100))

        plt.show()

    @staticmethod
    def check_bin(i, bin_x):
        '''
        Checks each bin to validate before OLS calculation.
        '''
        # Will calculate slope + intercept in 1st segment so need >2 xs
        if i == 0 and len(bin_x) < 2:
            raise ValueError('Need at least 2 data points in the 1st segment.')
        # Need len(bin) > 1 in all other bins
        if len(bin_x) < 1:
            raise ValueError('Need at least 1 data point in every segment but the 1st.')
        # Need a non-zero matrix
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
        + attributes from smoother class

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
    >>> model.fit(x, y, plot=True, confidence_interval=0.99)
    """
    def __init__(self, knots):
        linear_spline.__init__(self, knots)

    def fit(self, x, y, plot=False, confidence_interval=False):
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
        
        # Store residuals and smoothed values
        self.smoothed_ys = self.predict(x)
        self.residuals = y - self.smoothed_ys
    
        # Plot fit model
        if plot:
            self.plot_model(x, y, confidence_interval)

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

class bin_regression(smoother):
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
        + attributes from smoother class

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
    >>> model.fit(x, y, plot=True, confidence_interval=0.99)
    """

    def __init__(self, bins):
        smoother.__init__(self)
        self.bins = self.check_bins_input(bins) # number of bins
        self.bin_ys = np.zeros(bins) # store each bin y value
        self.bin_vals = None #set bin values model.fit()

    def fit(self, x, y, plot=False, confidence_interval=False):
        '''
        Given input arrays x and y. Fits the model.
        '''
        # Validate x and y input
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')

        self.bin_vals = np.linspace(x.min(), x.max(), num=self.bins+1)

        for i in range(self.bins):
            # Select bin ys and shift bin starting x and starting y to 0,0
            bin_y = y[np.logical_and(x>=self.bin_vals[i], x<=self.bin_vals[i+1]).flatten()]
            self.check_bin(i, bin_y)
            self.bin_ys[i] = np.mean(bin_y)

        # Store residuals and smoothed values
        self.smoothed_ys = self.predict(x)
        self.residuals = y - self.smoothed_ys
    
        # Plot model
        if plot:
            self.plot_model(x, y, confidence_interval)

    def predict(self, x):
        '''
        Given a 1-dimenional numpy array, make predictions.
        '''
        preds = np.zeros(len(x))
        # iterate over every x in input array
        for i in range(len(x)):
            broke = False
            if x[i] <= self.bin_vals[0]:
                preds[i] = self.bin_ys[0]
            else:
                # iterate until in correct bin
                for j in range(len(self.bin_vals)-1):
                    if x[i] >= self.bin_vals[j] and x[i] < self.bin_vals[j+1]:
                        preds[i] = self.bin_ys[j]
                        broke = True
                        break
                if broke == False:
                    preds[i] = self.bin_ys[-1]
        return preds
        
    def plot_model(self, x, y, confidence_interval=False):
        '''
        Plot the models hypothetical predictions, MSE, and true data points.
        '''
        # Plot each bin (left + right limit) + data points
        modelx = np.repeat(self.bin_vals, 2)[1:-1]
        modely = np.repeat(self.bin_ys, 2)
        plt.plot(modelx, modely, color='tab:orange')
        plt.scatter(x, y)

        # MSE + Title
        MSE = np.mean((self.residuals) ** 2)
        plt.title("{}, Bins: {}, MSE: {:.8f}".format(type(self).__name__, self.bins, MSE))

        # Optional: Plot confidence interval
        if confidence_interval:
            band_size = self.get_confidence_interval(confidence_interval)
            plt.fill_between(modelx, modely - band_size, modely + band_size, color='tab:blue', alpha=0.2)
            plt.title("{}, MSE: {:.4f}, Confidence Interval: {:.8g}%".format(type(self).__name__, MSE, confidence_interval*100))
        
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

class cubic_spline(smoother):
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
        + attributes from smoother class

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
    >>> model.fit(x, y, plot=True, confidence_interval=0.90)

    Reference
    ----------
    Kong, Qingkai, et al. Python Programming and Numerical Methods: A Guide for 
    Engineers and Scientists. Academic Press, an Imprint of Elsevier, 
    pythonnumericalmethods.berkeley.edu, Accessed July 2022. 
    """

    def __init__(self, pieces):
        smoother.__init__(self)
        self.pieces = self.check_input_pieces(pieces)
        self.knot_xvals = None
        self.knot_yvals = None
        self.ws = None # store weights for each piecewise polynomial

    def fit(self, x, y, plot=False, confidence_interval=False):
        '''
        Given input arrays x and y. Fits the model.
        '''
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')
        if len(x) <= self.pieces:
            raise ValueError('max number of pieces is len(x) - 1')

        # Values must be sorted for knot calculation
        y = y[x.argsort()]
        x = np.sort(x)

        # Find (x,y) for each knot value
        self.knot_xvals = np.linspace(x.min(), x.max(), num=self.pieces+1)
        self.knot_yvals = self.find_knots(self.knot_xvals, x, y)
        
        # Calculate weights of each piecewise function
        self.ws = self.calc_piecewise_weights(self.knot_xvals, self.knot_yvals)

        # Store residuals and smoothed values
        self.smoothed_ys = self.predict(x)
        self.residuals = y - self.smoothed_ys

        # Plot model
        if plot:
            self.plot_model(x, y, confidence_interval)

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
        # Array for predictions (assign is faster than append)
        ys = np.zeros(len(xs), dtype=np.float64) # use float for precision

        ws_pos = 0
        insert_pos = 0
        for i in range(self.pieces+1):
            # Add points before startpoint
            if i == 0:
                ys_to_add = self.polynomial(xs[xs<=self.knot_xvals[i]], self.ws[ws_pos:ws_pos+4][::-1])
                ys[:len(ys_to_add)] = ys_to_add
            # All other pieces
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

    def plot_model(self, x, y, confidence_interval=False):
        '''
        Plots the models hypothetical predictions, MSE, and true data points.
        '''
        # Plot hypothetical model values
        modelx = np.linspace(self.knot_xvals[0], self.knot_xvals[-1], self.pieces*10)
        modely = self.predict(modelx)
        plt.plot(modelx, modely, color='tab:orange')

        # Plot data values
        plt.scatter(x, y)
        
        # MSE + Title
        MSE = np.mean(self.residuals ** 2)
        plt.title("{}, Pieces: {}, MSE: {:.8f}".format(type(self).__name__, self.pieces, MSE))
        
        # Optional: Plot confidence interval
        if confidence_interval:
            band_size = self.get_confidence_interval(confidence_interval)
            plt.fill_between(modelx, modely - band_size, modely + band_size, color='tab:blue', alpha=0.2)
            plt.title("{}, MSE: {:.4f}, Confidence Interval: {:.8g}%".format(type(self).__name__, MSE, confidence_interval*100))
        
        plt.show()

    @staticmethod
    def find_knots(knot_vals, xs, ys):
        '''
        Helper function to find the y-value for each knot in xs.
        '''
        knot_ys = np.zeros(len(knot_vals))
        for i, knot in enumerate(knot_vals):
            # If knot is a data point, uses data point
            if knot in xs:
                knot_ys[i] = ys[np.where(xs == knot)[0][0]]
            
            # If not, take the weighted average of closest points
            else:
                # Getting first element larger than the knot
                index = np.where(xs > knot)[0][0]
                
                # Getting x-distance between closest points 
                ld = knot - xs[index-1]
                rd = xs[index] - knot

                # Calculate weighted average
                knot_ys[i] = np.average(ys[index-1:index+1], weights=[rd, ld])
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
        + attributes from smoother class

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
    >>> model.fit(x, y, plot=True, confidence_interval=0.95)
    """
    def __init__(self, pieces):
        cubic_spline.__init__(self, pieces)

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
                # Calculate values with a shift based on starting knot xvalue
                ys_to_add = self.line(xs[xs<=self.knot_xvals[i]] - self.knot_xvals[0], slope, p2)
                ys[:len(ys_to_add)] = ys_to_add

            # all other pieces
            else:
                ys_to_add = self.polynomial(xs[np.logical_and(xs>self.knot_xvals[i-1], xs<=self.knot_xvals[i])], self.ws[ws_pos:ws_pos+4][::-1])
                ws_pos+=4
            
            # insert predictions to array, moving pointer
            ys[insert_pos:insert_pos + len(ys_to_add)] = ys_to_add
            insert_pos += len(ys_to_add)

        # Add points after endpoint
        ws_pos-=4
        
        # Naive calculation of slope
        p1 = self.polynomial(self.knot_xvals[-1:], self.ws[ws_pos:ws_pos+4][::-1])
        p2 = self.polynomial(self.knot_xvals[-1:] + 0.01, self.ws[ws_pos:ws_pos+4][::-1])
        slope = (p2 - p1) / 0.01
        ys_to_add = self.line(xs[xs>self.knot_xvals[i]] - self.knot_xvals[-1], slope, p2)
        
        # insert points past the endpoint to predictions array
        ys[insert_pos:insert_pos + len(ys_to_add)] = ys_to_add
        return ys
    
    @staticmethod
    def line(x, slope, intercept):
        return (slope*x) + intercept

class exp_moving_average(smoother):
    """Exponential moving average.

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
        + attributes from smoother class

    Raises
    ---------
        TypeError: if alpha is not a floating point
        ValueError: if alpha is not >0 and <1

    Example
    ---------
    >>> from regressio.models import exp_moving_average
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(200)
    >>> model = exp_moving_average(alpha=0.1)
    >>> model.fit(x, y, plot=True, confidence_interval=0.99)

    Reference
    ----------
    Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 
    3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed July 2022.
    """
    
    def __init__(self, alpha=0.2):
        smoother.__init__(self)
        self.alpha = self.check_alpha(alpha)

    def fit(self, x, y, plot=False, confidence_interval=False):
        '''
        Given arrays x and y, computes exponentially smoothed y values.
        '''
        # Validate x and y input
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')
        if len(y) <= 2:
            raise ValueError('len(y) must be greater than 2')
        
        # Array for smoothed values
        smoothed_ys = np.zeros_like(y)
        smoothed_ys[0] = np.mean(y[:10]) # Initialize initial value using Mean

        # Iterate over y values
        for i in range(1, len(y)):
            smoothed_ys[i] = self.alpha*y[i-1] + (1-self.alpha)*smoothed_ys[i-1]

        # Store residuals and smoothed values
        self.smoothed_ys = smoothed_ys
        self.residuals = y - smoothed_ys

        if plot:
            self.plot_model(x, y, confidence_interval)

    def plot_model(self, x, y, confidence_interval=False):
        '''
        Plots the smoothed model, MSE, and true data points and an 
        optional confidence interval.
        '''
        # Plot smoothed model values
        plt.plot(x, self.smoothed_ys, color='tab:orange')
        
        # Plot actual data values
        plt.scatter(x, y)
        
        # MSE and Title
        MSE = np.mean(self.residuals ** 2)
        plt.title("{}, MSE: {:.8f}".format(type(self).__name__, MSE))
        
        # Optional param: plot confidence interval
        if confidence_interval:
            band_size = self.get_confidence_interval(confidence_interval)
            plt.fill_between(x, self.smoothed_ys - band_size, self.smoothed_ys + band_size, color='tab:blue', alpha=0.2)
            plt.title("{}, MSE: {:.4f}, Confidence Interval: {:.8g}%".format(type(self).__name__, MSE, confidence_interval*100))
        
        plt.show()

    @staticmethod
    def check_alpha(alpha):
        '''
        Validate alpha input.
        '''
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha < 0 or alpha > 1:
            raise ValueError('alpha must be between 0 and 1')
        return alpha

class gaussian_kernel(smoother):
    """Gaussian kernel.

    Kernel smoothing: For each value in a given array, each smoothed value is 
    calculated as some function applied to the original value and its surrounding points. 
    
    For the gaussian kernel, we center the gaussian distribution around each point and
    take the sum of the weighted values over all the data. The smoothness of the function is
    tuned by the full width half maximum parameter. For a complete description of the kernel 
    function see the reference below.

    Args
    ---------
        fwhm: width of kernel at 1/2 max height of the gaussian distribution

    Attributes
    ---------
        fwhm: width of kernel at 1/2 max height of the gaussian distribution
        + attributes from smoother class

    Raises
    ---------
        TypeError: if fwhm is not a float or int

    Example
    ---------
    >>> from regressio.models import gaussian_kernel
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(100)
    >>> model = gaussian_kernel(fwhm=4)
    >>> model.fit(x, y, plot=True, confidence_interval=0.90)

    Reference
    ----------
    Brett, M. (2014, October 26). An introduction to smoothing. 
    Tutorials on imaging, computing and mathematics. matthew-brett.github.io/teaching, 
    Accessed July 2022.
    """
    def __init__(self, fwhm=None):
        smoother.__init__(self)
        self.fwhm = self.check_fwhm(fwhm)

    def fit(self, x, y, plot=False, confidence_interval=False):
        """
        Given arrays x and y, computes smoothed y values.
        """
        # Validate x and y input
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')

        # Initiliaze smoothed value array
        self.smoothed_ys = np.zeros(y.shape)
    
        # Sort array for kernel + FWHM calculation
        y = y[x.argsort()]
        x = np.sort(x)

        # Set FWHM + calculate sigma
        if self.fwhm == None:
            self.fwhm = (x[-1] - x[0]) / 25
        sigma = self.fwhm_to_sigma(self.fwhm)

        # Calculate kernel at each point
        for i, x_position in enumerate(x):
            kernel = self.kernel_at_position(x, x_position, sigma)
            kernel = kernel / sum(kernel)
            self.smoothed_ys[i] = sum(y * kernel)
        
        # Storing residuals
        self.residuals = y - self.smoothed_ys

        # Plot model
        if plot:
            self.plot_model(x, y, confidence_interval)

    def plot_model(self, x, y, confidence_interval=False):
        '''
        Plots the models hypothetical predictions, MSE, and true data points.
        '''
        # Plot model + data points
        plt.plot(x, self.smoothed_ys, color='tab:orange')
        plt.scatter(x, y)

        # MSE + title
        MSE = np.mean((self.residuals) ** 2)
        plt.title("{}, FWHM: {}, MSE: {:.4f}".format(type(self).__name__, self.fwhm, MSE))

        # Optional: Plot confidence interval
        if confidence_interval:
            band_size = self.get_confidence_interval(confidence_interval)
            plt.fill_between(x.flatten(), self.smoothed_ys - band_size, self.smoothed_ys + band_size, color='tab:blue', alpha=0.2)
            plt.title("{}, FWHM: {}, MSE: {:.4f}, Confidence Interval: {:.8g}%".format(type(self).__name__, self.fwhm, MSE, confidence_interval*100))
        plt.show()

    def fwhm_to_sigma(self, fwhm):
        """
        The FWHM is the full width of the kernel at half the maximum height 
        of the Gaussian function. For a Gaussian function with standard deviation 
        1, the maximum height is ~0.4. The width of the kernel at 0.2 to 0.2 (on the Y axis) 
        is the FWHM.

        This function takes in a FWHM value and returns sigma (a standard deviation).

        Formula: https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        """
        return fwhm / (2 * np.sqrt(2 * np.log(2)))

    def kernel_at_position(self, x, x_position, sigma):
        """
        At a given index denoted x_position, returns the kernel 
        weights over array x.

        Formula: https://en.wikipedia.org/wiki/Gaussian_filter
        """
        kernel = np.exp(-(x - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        return kernel

    @staticmethod
    def check_fwhm(fwhm):
        '''
        Validate fwhm input.
        '''
        if fwhm == None:
            return None
        if type(fwhm) != float and type(fwhm) != int:
            raise TypeError('fwhm must be a float or int')
        return fwhm

class knn_kernel(smoother):
    """KNN kernel. 
    
    The KNN kernel is applied to each data point. The smoothed value is
    the average of the N nearest points. The smoothness of the function is
    determined by the size of N. The larger N is, the smoother the function.

    If the boundary of the training data is reached before we get to N
    data points, then we compute the smoothed values with less than N
    points. This results in a smooth fitting line.

    Args
    ---------
        n: the number of nearest points used in KNN calculation

    Attributes
    ---------
        n: the number of nearest points used in KNN calculation
        + attributes from smoother class

    Raises
    ---------
        ValueError: if n is <= 0 | n >= len(x)
        TypeError: if n is not an integer

    Example
    ---------
    >>> from regressio.models import knn_kernel
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(100)
    >>> model = knn_kernel(n=6)
    >>> model.fit(x, y, plot=True, confidence_interval=0.90)
    """
    def __init__(self, n):
        smoother.__init__(self)
        self.n = self.check_n_input(n)

    def fit(self, x, y, plot=False, confidence_interval=False):
        """
        Given arrays x and y, computes smoothed y values.
        """
        # Validate x and y input
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')
        if self.n > len(x):
            raise ValueError('max N value is len(x)')
    
        # Sort array for kernel calculation
        y = y[x.argsort()]
        x = np.sort(x)

        # Calculate smoothed ys + storing residuals
        self.smoothed_ys = self.find_closest_n_points(x, y)
        self.residuals = y - self.smoothed_ys

        # Plot model
        if plot:
            self.plot_model(x, y, confidence_interval)

    def plot_model(self, x, y, confidence_interval=False):
        '''
        Plots the models hypothetical predictions, MSE, and true data points.
        '''
        # Plot model + data points
        plt.plot(x, self.smoothed_ys, color='tab:orange')
        plt.scatter(x, y)

        # MSE + title
        MSE = np.mean((self.residuals) ** 2)
        plt.title("{}, n: {}, MSE: {:.4f}".format(type(self).__name__, self.n, MSE))

        # Optional Param: Plot confidence interval
        if confidence_interval:
            band_size = self.get_confidence_interval(confidence_interval)
            plt.fill_between(x.flatten(), self.smoothed_ys - band_size, self.smoothed_ys + band_size, color='tab:blue', alpha=0.2)
            plt.title("{}, n: {}, MSE: {:.4f}, Confidence Interval: {:.8g}%".format(type(self).__name__, self.n, MSE, confidence_interval*100))
        plt.show()

    def find_closest_n_points(self, x, y):
        """
        Calculates KNN for all points in array y given x and y.
        """
        smoothed_values = np.zeros(len(y))

        # Calculate kernel for every x_value
        for i, x_value in enumerate(x):

            closest_points = [0] * self.n
            pi = 0
            li = np.searchsorted(x, x_value)
            broke = False

            # Set starting point for li, ri in array
            if li == 0:
                ri = 1
            elif li == len(x) - 1:
                ri = li
                li -= 1
            else:
                if abs(x_value - x[li]) < abs(x_value - x[li+1]):
                    ri = li
                    li -= 1
                else:
                    ri = li + 1

            # Find up to the closest n points
            for j in range(self.n):
                if li >= 0 and ri < len(x):
                    if abs(x_value - x[li]) < abs(x_value - x[ri]):
                        closest_points[pi] = y[li]
                        li -= 1
                    else:
                        closest_points[pi] = y[ri]
                        ri += 1
                # If border reached, return current closest values,
                # otherwise smoothed values at edges are horizontal lines
                else:
                    broke = True
                    break
                pi+=1

            if broke:
                smoothed_values[i] = np.mean(closest_points[:j])
            else:
                smoothed_values[i] = np.mean(closest_points[:j+1])

        return smoothed_values

    @staticmethod
    def check_n_input(n):
        '''
        Validate n parameter.
        '''
        if type(n) != int:
            raise TypeError('n must be an integer')
        if n < 1:
            raise ValueError('n must be >= 1')
        return n

class weighted_average_kernel(smoother):
    """Weighted average kernel. 
    
    The Weighted average kernel is applied to each data point. The smoothed value is
    the weighted average of points within a specified distance. The smoothness of the 
    function is determined by the size of the distance. The larger the distance, the
    smoother the line.

    Args
    ---------
        dist: the max distance for points to be in order to be in kernel

    Attributes
    ---------
        dist: the max distance for points to be in order to be in kernel
        + attributes from smoother class

    Raises
    ---------
        ValueError: if dist is <= 0
        TypeError: if dist is not an integer

    Example
    ---------
    >>> from regressio.models import weighted_average_kernel
    >>> from regressio.datagen import generate_random_walk
    >>> x, y = generate_random_walk(100)
    >>> model = weighted_average_kernel(dist=6)
    >>> model.fit(x, y, plot=True, confidence_interval=0.90)
    """
    def __init__(self, dist):
        smoother.__init__(self)
        self.dist = self.check_dist_input(dist)

    def fit(self, x, y, plot=False, confidence_interval=False):
        """
        Given arrays x and y, computes smoothed y values.
        """
        # Validate x and y input
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')

        # Checking that self.dist is not too small
        largest_dist = abs(x[1] - x[0])
        for i in range(len(x) - 1):
            if abs(x[i+1] - x[i]) > largest_dist:
                largest_dist = abs(x[i+1] - x[i])

        if self.dist < largest_dist:
            raise ValueError('dist must be >= {} for this dataset'.format(largest_dist))
    
        # Sort array for kernel calculation
        y = y[x.argsort()]
        x = np.sort(x)

        # Calculate smoothed ys + storing residuals
        self.smoothed_ys = self.find_close_points(x, y)
        self.residuals = y - self.smoothed_ys

        # Plot model
        if plot:
            self.plot_model(x, y, confidence_interval)

    def plot_model(self, x, y, confidence_interval=False):
        '''
        Plots the models hypothetical predictions, MSE, and true data points.
        '''
        # Plot model + data points
        plt.plot(x, self.smoothed_ys, color='tab:orange')
        plt.scatter(x, y)

        # MSE + title
        MSE = np.mean((self.residuals) ** 2)
        plt.title("{}, dist: {}, MSE: {:.4f}".format(type(self).__name__, self.dist, MSE))

        # Optional Param: Plot confidence interval
        if confidence_interval:
            band_size = self.get_confidence_interval(confidence_interval)
            plt.fill_between(x.flatten(), self.smoothed_ys - band_size, self.smoothed_ys + band_size, color='tab:blue', alpha=0.2)
            plt.title("{}, dist: {}, MSE: {:.4f}, Confidence Interval: {:.8g}%".format(type(self).__name__, self.dist, MSE, confidence_interval*100))
        plt.show()

    def find_close_points(self, x, y):
        """
        Calculates weighted average for all points in array y given x and y.
        """

        # Smoothed values, and weighted closed points storage
        smoothed_values = np.zeros(len(x))
        closest_points = np.zeros(len(x))
        cp_weights = np.zeros(len(x))

        # Calculate kernel for every x_value
        for i, x_value in enumerate(x):
            pi = 0
            li = np.searchsorted(x, x_value)
            ri = li
            broke = False
            
            # Set starting point for li, ri in array
            if li == 0:
                ri = 1
            elif li == len(x) - 1:
                ri = li
                li -= 1
            else:
                if abs(x_value - x[li]) < abs(x_value - x[li+1]):
                    ri = li
                    li -= 1
                else:
                    ri = li + 1
            
            # Find all points within +- dist of point (similar to merge sort method)
            for j in range(len(x)):
                if li >= 0 and ri < len(x):
                    if min(abs(x_value - x[li]), abs(x_value - x[ri])) <= self.dist:
                        if abs(x_value - x[li]) < abs(x_value - x[ri]):
                            closest_points[pi] = y[li]
                            cp_weights[pi] = abs(x_value - x[li])
                            li -= 1
                        else:
                            closest_points[pi] = y[ri]
                            cp_weights[pi] = abs(x_value - x[ri])
                            ri += 1
                    else:
                        broke = True
                        break
                elif li >= 0 and abs(x_value - x[li]) <= self.dist:
                    closest_points[pi] = y[li]
                    cp_weights[pi] = abs(x_value - x[li])
                    li -= 1
                elif ri < len(x) and abs(x_value - x[ri]) <= self.dist:
                    closest_points[pi] = y[ri]
                    cp_weights[pi] = abs(x_value - x[ri])
                    ri += 1
                else:
                    broke = True
                    break

                pi+=1

            # Workaround to solve division by 0 error
            cp_weights[:j] = np.reciprocal(cp_weights[:j]+1)
            
            if broke:
                smoothed_values[i] = np.average(closest_points[:j], weights=cp_weights[:j])
            else:
                smoothed_values[i] = np.average(closest_points[:j+1], weights=cp_weights[:j+1])

        return smoothed_values

    @staticmethod
    def check_dist_input(dist):
        '''
        Validate dist parameter.
        '''
        if type(dist) != int:
            raise TypeError('dist must be an integer')
        if dist <= 0:
            raise ValueError('dist must be >= 0')
        return dist

if __name__ == '__main__':
    main()