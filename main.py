from regressio.datagen import generate_random_walk
from regressio.models import polynomial_regression

def main():
    x, y = generate_random_walk(100, 1)
    model = polynomial_regression(10)
    model.fit(x, y)
    return
    
if __name__ == '__main__':
    main()