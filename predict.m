function p = predict(Theta1, Theta2, X)
%predict the label of an input given a trained neural network
%based on code from Andrew Ng Coursera course on Machine Learning

m = size(X, 1);
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
p=(h2>=0.5);

end
