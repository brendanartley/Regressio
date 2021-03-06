import pytest
import numpy as np
from regressio.datagen import generate_random_walk
from regressio.models import *

'''
Pytest:
- prefix your class with 'Test', and functions with 'test' otherwise they will be skipped.
'''

class Test_generate_random_walk: 
    def test_positive(self):
        x, y = generate_random_walk(10, 1)
        assert len(x) == len(y) == 10

    def test_zero(self):
        with pytest.raises(ValueError):
            x, y = generate_random_walk(0, 1)

    def test_negative(self):
        with pytest.raises(ValueError):
            x, y = generate_random_walk(-10, -1)

class Test_linear_regression: 
    def test_degree_greater_than_10(self):
        with pytest.raises(ValueError):
            model = linear_regression(degree=11)

    def test_degree_less_than_10(self):
        with pytest.raises(ValueError):
            model = linear_regression(degree=-1)
    
    def test_invalid_input(self):
        with pytest.raises(TypeError):
            model = linear_regression(degree='0')
    
    def test_fit_model(self):
        x, y = generate_random_walk(100)
        model = linear_regression(degree=10)
        model.fit(x, y)
            
class Test_linear_spline: 
    def test_no_data_in_knot(self):
        with pytest.raises(ValueError):
            x, y = np.arange(20), np.cumsum(np.ones(20))
            model = linear_spline(21)
            np.random.shuffle(x)
            model.fit(x, y)

    def test_less_than_two_knots(self):
        with pytest.raises(ValueError):
            model = linear_spline(knots=1)

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            model = linear_spline(knots='0')

    def test_fit_model(self):
        x, y = generate_random_walk(100)
        model = linear_spline(knots=12)
        model.fit(x, y)

class Test_isotonic_regression: 
    def test_no_data_in_knot(self):
        with pytest.raises(ValueError):
            x, y = np.arange(20), np.cumsum(np.ones(20))
            model = isotonic_regression(21)
            model.fit(x, y)
    
    def test_invalid_input(self):
        with pytest.raises(TypeError):
            model = isotonic_regression(knots='0')

    def test_fit_model(self):
        x, y = generate_random_walk(100)
        model = isotonic_regression(knots=12)
        model.fit(x, y)

class Test_bin_regression: 
    def test_no_data_in_knot(self):
        with pytest.raises(ValueError):
            x, y = np.arange(20), np.cumsum(np.ones(20))
            model = bin_regression(bins=21)
            model.fit(x, y)

    def test_less_than_one_bin(self):
        with pytest.raises(ValueError):
            model = bin_regression(bins=0)

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            model = bin_regression(bins='0')

    def test_fit_model(self):
        x, y = generate_random_walk(100)
        model = bin_regression(bins=8)
        model.fit(x, y)

class Test_cubic_spline: 
    def test_degree_less_than_2(self):
        with pytest.raises(ValueError):
            model = cubic_spline(pieces=1)
    
    def test_invalid_input(self):
        with pytest.raises(TypeError):
            model = cubic_spline(pieces='3')

    def test_data_too_small(self):
        with pytest.raises(ValueError):
            x, y = generate_random_walk(10)
            model = cubic_spline(pieces=len(x))
            model.fit(x,y)
    
    def test_fit_model(self):
        x, y = generate_random_walk(100)
        model = cubic_spline(pieces=50)
        model.fit(x, y)

class Test_natural_cubic_spline: 
    def test_fit_model(self):
        x, y = generate_random_walk(100)
        model = natural_cubic_spline(pieces=50)
        model.fit(x, y)

class Test_exponential_smoother: 
    def test_zero_alpha(self):
        with pytest.raises(TypeError):
            model = exponential_smoother(alpha=0)
    
    def test_negative_alpha(self):
        with pytest.raises(ValueError):
            model = exponential_smoother(alpha=-1.2)

    def test_data_too_small(self):
        with pytest.raises(ValueError):
            x, y = generate_random_walk(2)
            model = exponential_smoother()
            model.fit(x,y)