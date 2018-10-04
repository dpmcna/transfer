function g = sigmoid(z)
%compute sigmoid functoon
%based on code from Andrew Ng Coursera course on Machine Learning

g = 1.0 ./ (1.0 + exp(-z));

end
