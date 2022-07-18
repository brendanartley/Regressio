import numpy as np
import matplotlib.pyplot as plt

class linear_regression():
    '''
    Polynomial regression model up to the 10th degree. 
    - Degree > 10 is numerically unstable in OLS calculation.
    '''
    def __init__(self, degree):
        self.degree = self.check_degree(degree)
        self.ws = np.random.random(size=degree).astype(np.float128)
        self.range = [-1,1]
        self.rmse = None

    @staticmethod
    def check_degree(value):
        '''
        Validates INT input and degree < 10. Model is unstable for degree > 10
        '''
        if not isinstance(value, int):
            raise TypeError('Invalid Type. Int type Expected')
        if value < 0 or value > 10:
            raise ValueError('0 <= degree <= 10. Model is unstable for degree > 10.')
        return value

    def fit(self, rawx, y, plot=False):
        '''
        Given input arrays x and y. Fits the model.
        '''
        # -- Validate input data -- 
        if len(rawx) != len(y):
            raise ValueError('x and y must be of the same length')
        elif len(rawx) <= 1:
            raise ValueError('len(x) must be > 1')

        # Stores min, max of x for plot_polynomial function
        self.range = [rawx.min(), rawx.max()]

        # -- Calculate ordinary least squares --
        x = np.hstack([(rawx**i) for i in range(self.degree+1)])
        xTx = x.T.dot(x)
        xTx_inv = np.linalg.inv(xTx)
        ws = xTx_inv.dot(x.T.dot(y))
        self.ws = ws # storing model weights
        
        # -- Calculating MSE --
        training_mse = self.MSE(x, y)

        # -- Store model weights and plot results -- 
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
        # -- Creates model prediction line -- 
        modelx = np.linspace(self.range[0], self.range[1], 1000).reshape([-1, 1])
        expanded = np.hstack([self.ws[i] * (modelx ** i) for i in range(0, len(self.ws))])
        modely = np.sum(expanded, axis=-1)

        # -- Plot training data
        plt.scatter(x, y)
        
        # -- Plots model prediction line --
        plt.scatter(modelx, modely, s=5, color='tab:orange')
        plt.title("Linear Regression, Degree: {}, MSE: {:.8f}".format(self.degree, training_mse))
        plt.show()

class isotonic_regression():
    '''
    Isotonic regression model. 
    '''
    def __init__(self, knots):
        self.knots = knots # number of kknots
        self.slopes = np.zeros(knots) # stores piecewise linear model slopes
        self.last_binvals = np.zeros(knots+1) # stores last y-value in each bin
        self.bins = None #set bin values model.fit()

    
    def fit(self, x, y, plot=False):
        '''
        Given input arrays x and y. Fits the model.
        '''
        # check inputs are the same length
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')

        ysum = y.min()
        last_binval = y.min()
        self.last_binvals[0] = y[:10].min()
        self.bins = np.linspace(x.min(), x.max(), num=self.knots+1)

        for i in range(self.knots):
            # Select points in the current bin
            bin_x = x[np.logical_and(x>self.bins[i], x<=self.bins[i+1])]
            if len(bin_x) < 1: # >= 1 data point in each bin.
                raise ValueError('Need at least 1 data point in each segment')

            # Select bin ys and shift bin starting x and starting y to 0,0
            bin_y = y[np.logical_and(x>self.bins[i], x<=self.bins[i+1]).flatten()]
            bin_y = bin_y - last_binval
            bin_x = bin_x - self.bins[i]

            # Find best fit line for the bin
            slope = self.ISO_OLS(bin_x, bin_y)
            self.slopes[i] = slope

            # Store the last value in the bin (to add onto next bin)
            last_binval = self.line(self.bins[1] - self.bins[0], slope, last_binval)
            self.last_binvals[i+1] = last_binval
            ysum += last_binval
    
        if plot:
            # plot fit model
            self.plot_model(x, y)

    def predict(self, x):
        '''
        Given a 1-dimenional numpy array, makes predictions.
        '''
        preds = np.zeros(len(x))
        # iterating every x in input array
        for i in range(len(x)):
            broke = False
            x_raw = x[i] - self.bins[0]
            # if less than starting bin set to starting value
            if x[i] <= self.bins[0]:
                preds[i] = self.last_binvals[0]
                continue
            else:
                # iterating over every bin
                for j in range(len(self.bins)-1):
                    if x[i] > self.bins[j] and x[i] < self.bins[j+1]:
                        preds[i] = self.line(x_raw, self.slopes[j], self.last_binvals[j])
                        broke = True
                        break
                    # shift x until its bin starts at origin
                    else:
                        x_raw -= self.bins[1] - self.bins[0]
                # if x > last_bin set to last_binval
                if broke == False:
                    preds[i] = self.last_binvals[j+1]
        return preds
    
    @staticmethod
    def line(x, slope, intercept):
        '''
        Simple line function.
        '''
        return slope*x + intercept

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
    
    def plot_model(self, x, y):
        '''
        Plots the models hypothetical predictions, MSE, and true data points.
        '''
        plt.plot(self.bins, self.last_binvals, color='tab:orange')
        plt.scatter(x, y)

        preds = self.predict(x)
        MSE = np.mean((y - preds) ** 2)
        
        plt.title("Isotonic Regression, Knots: {}, MSE: {:.8f}".format(self.knots, MSE))

        for i in range(len(self.bins)):
            plt.axvline(x = self.bins[i], alpha=0.2)
        plt.show()

if __name__ == '__main__':
    main()