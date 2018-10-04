function g = sigmoidGradient(z)
%returns the gradient of the sigmoid function evaluated at z
%based on code from Andrew Ng Coursera course on Machine Learning

g=sigmoid(z).*(1-sigmoid(z));

end
