function p = predictOneVsAll(all_theta, X)
%predict the label for a trained one-vs-all logistic classifier
%based on code from Andrew Ng Coursera course on Machine Learning

m = size(X, 1);
X = [ones(m, 1) X]; 
preds=sigmoid(X*all_theta');
p=(preds>=0.5);

end
