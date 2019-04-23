# ML-CarMPGPredictor

Use gradient descent algorithm to determine a relationship between the miles per gallon and horsepower of a car using the Tensorflow library. This follows the "Machine Learning with Javascript" course from Udemy.

Miles per Gallon = m \* (Car Horsepower) + b

### class LinearRegression with methods:

- gradientDescent() - Run one iteration of GD and update 'm' and 'b'
- train() - Run GD until we get good values for 'm' and 'b'
- test() - Use 'test' data set to evaluate the accuracy of our calculated 'm' and 'b'
- predict() - Make a prediction using our calculated 'm' and 'b'

### How Gradient Descent Works

- Pick a value for 'b' and 'm'
- Calculate the slope of MSE with respect to 'm' and 'b'
- Are both slopes very small? If so, we are done.
- Multiply both slopes by learning rate
- Subtract results from 'b' and 'm'
