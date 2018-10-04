function [J, grad] = lrCostFunction(theta, X, y, lambda)
%compute cost and gradient for logistic regression with regularization
%based on code from Andrew Ng Coursera course on Machine Learning

m = length(y); 
J = 1/m * sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)));
grad = 1/m*X'*(sigmoid(X*theta)-y);
temp = theta; 
temp(1) = 0; 
J = J + lambda/(2*m) * sum(temp.^2);
grad = grad + lambda/m * temp;
grad = grad(:);

end
