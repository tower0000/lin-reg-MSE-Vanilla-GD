# Gradient descent
Code implementing the gradient descent to minimize MSE function of linear regression model to find optimal model weights.

Mean squared error:

$MSE = \frac{1}{n}\sum_{(y_i-X_iw)^2}$

Gradient of the mse function:

$\frac{∂ MSE}{∂ w} = \frac{1 \cdot 2}{n}({Y - Xw}) \cdot-X$

The number of model features is selected in line 34 (and in line 63 for an example of work)
