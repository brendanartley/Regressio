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
        # -- Validates Degree Input -- 
        if not isinstance(value, int):
            raise TypeError('Invalid Type. Int type Expected')
        if value < 0 or value > 10:
            # -- Degree > 10 is numerically unstable
            raise ValueError('1 <= degree <= 10. Model is unstable for degree > 10.')
        return value

    def fit(self, rawx, y):
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
        self.plot_model(rawx, y, training_mse)

    def predict(self, x):
        values = np.hstack([self.ws[i] * (x ** i) for i in range(0, len(self.ws))])
        preds = np.sum(values, axis=-1)
        return preds

    def MSE(self, x, y):
        y_hat = x.dot(self.ws)
        loss = np.mean((y - y_hat) ** 2)
        return loss

    def plot_model(self, x, y, training_mse):
        # -- Creates model prediction line -- 
        modelx = np.linspace(self.range[0], self.range[1], 1000).reshape([-1, 1])
        expanded = np.hstack([self.ws[i] * (modelx ** i) for i in range(0, len(self.ws))])
        modely = np.sum(expanded, axis=-1)

        # -- Plot training data
        plt.scatter(x, y)
        
        # -- Plots model prediction line --
        plt.scatter(modelx, modely, s=5, color='tab:orange')
        plt.title("Degree: {}, Training MSE: {:.8f}".format(self.degree, training_mse))
        plt.show()

class isotonic_regression():
    '''
    Isotonic regression model. 
    '''
    def __init__(self, breakpoints):
        self.breakpoints = breakpoints
        self.slopes = np.zeros(breakpoints)
        self.last_binvals = np.zeros(breakpoints+1)
        self.bins = None #set this on model.fit()

    
    def fit(self, x, y):
        # -- Validate input data -- 
        if len(x) != len(y):
            raise ValueError('x and y must be of the same length')
        # need to add check if there is a training data point in each break point

        ysum = np.float64(0)
        last_binval = np.float64(0)
        self.bins = np.linspace(x.min(), x.max(), num=self.breakpoints+1)

        for i in range(self.breakpoints):
            bin_x = x[np.logical_and(x>self.bins[i], x<=self.bins[i+1])]
            if len(bin_x) < 1:
                raise ValueError('Need at least 1 data point in each segment')

            bin_y = y[np.logical_and(x>self.bins[i], x<=self.bins[i+1]).flatten()]
            bin_y = bin_y - last_binval
            bin_x = bin_x - self.bins[i]

            slope = self.OLS(bin_x, bin_y)
            self.slopes[i] = slope

            last_binval = self.line(self.bins[1] - self.bins[0], slope, last_binval)
            self.last_binvals[i] = last_binval
            ysum += last_binval

        self.plot_model(x, y)

    def predict(self, x):
        preds = np.zeros(len(x))
        for i in range(len(x)):
            broke = False
            x_raw = x[i]
            for j in range(len(self.bins)-1):
                if x[i] > self.bins[j] and x[i] < self.bins[j+1]:
                    preds[i] = self.line(x_raw, self.slopes[j], self.last_binvals[j-1])
                    broke=True
                    break
                else:
                    x_raw -= self.bins[1] - self.bins[0]
            if not broke:
                preds[i] = self.line(x_raw, self.slopes[j], self.last_binvals[j])
        return preds
    
    @staticmethod
    def line(x, slope, intercept):
        return slope*x + intercept

    @staticmethod
    def OLS(x, y):
        x = x.reshape(-1,1)
        xTx = x.T.dot(x)
        xTx_inv = np.linalg.inv(xTx)
        slope = xTx_inv.dot(x.T.dot(y))
        return max(np.asarray([0]), slope)
    
    def plot_model(self, x, y):
        # Technically I may be able to just plot breakpoints and connect them.
        modelx = np.arange(0.01, x.max(), 0.25)
        modely = self.predict(modelx)

        plt.scatter(modelx, modely, s=5, color='tab:orange')
        plt.scatter(x, y)

        for i in range(len(self.bins)):
            plt.axvline(x = self.bins[i], alpha=0.2)
        plt.show()
        
    
if __name__ == '__main__':
    main()