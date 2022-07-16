import numpy as np
import matplotlib.pyplot as plt

class polynomial_regression():
    '''
    Polynomial regression model up to the 10th degree. 
    - Degree > 10 is numerically unstable in OLS calculation.
    '''
    def __init__(self, degree):
        self.degree = self.check_degree(degree)
        self.ws = None
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
        if self.ws is None:
            raise ValueError('Must fit the model first.')
        values = np.hstack([self.ws[i] * (x ** i) for i in range(0, len(self.ws))])
        preds = np.sum(values, axis=-1)
        return preds

    def MSE(self, x, y):
        y_hat = x.dot(self.ws)
        loss = np.mean((y - y_hat) ** 2)
        return loss

    def plot_model(self, x, y, training_mse):
        # -- Creates model prediction line -- 
        x_values = np.linspace(self.range[0], self.range[1], 1000).reshape([-1, 1])
        expanded = np.hstack([self.ws[i] * (x_values ** i) for i in range(0, len(self.ws))])
        y_values = np.sum(expanded, axis=-1)
        
        # -- Plots model prediction line --
        plt.plot(x_values, y_values, color='tab:orange', linewidth=2, label='Model', alpha=0.75)
        plt.title("Degree: {}, Training MSE: {:.8f}".format(self.degree, training_mse))

        # -- Plot training data
        plt.scatter(x, y)
        plt.show()
    
if __name__ == '__main__':
    main()