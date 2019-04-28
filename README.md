# ML-CarMPGPredictor

Use gradient descent algorithm to determine a relationship between the miles per gallon and horsepower of a car using the Tensorflow library. This follows the "Machine Learning with Javascript" course from Udemy.

Miles per Gallon = m \* (Car Horsepower) + b

### class LinearRegression with methods:

- gradientDescent() - Run one iteration of GD and update 'm' and 'b'
- train() - Run GD until we get good values for 'm' and 'b'
- test() - Use 'test' data set to evaluate the accuracy of our calculated 'm' and 'b'
- predict() - Make a prediction using our calculated 'm' and 'b'

### How Gradient Descent Works

- Pick a value for 'b' and 'm' (and m2, m3, m4, etc.)
- Calculate the slope of MSE with respect to 'm' and 'b' using all observations
  - In batch gradient descent, you only use a portion of the observations
  - In a Stochastic Gradient Descent use one observation to calculate the slope. Just one row at a time. This way you come to a convergence much quicker.
- Are both slopes very small? If so, we are done.
- Multiply both slopes by learning rate
- Subtract results from "b" and "m"

### Custom Learning Rate Optimizer

- With every iteration of GD, calculate the exact value of MSE and store it
- After running an iteration of GD, look at the current MSE and the old MSE
- If the MSE went _up_ then we did a bad update, so divide learning rate by 2
- If the MSE went _down_ then we are going in the right direction! Increase LR by 5%
