function [all_theta] = oneVsAll(X, y, num_labels, lambda, max_iter)
%trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%based on code from Andrew Ng Coursera course on Machine Learning

m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1);
X = [ones(m, 1) X];
options = optimset('GradObj', 'on', 'MaxIter', max_iter);

for i = 1:num_labels
    initial_theta = zeros(n + 1, 1);
    c = i;
    [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)),initial_theta, options);
    all_theta(i,:)=theta';
end

end
