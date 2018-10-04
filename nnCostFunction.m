function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%neural network cost function
%based on code from Andrew Ng Coursera course on Machine Learning
                               
%extract lower level weights (Theta1) and upper level weights (Theta2)                               
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%compute neural network predictions using weights
m = size(X, 1);
X = [ones(m, 1) X];
a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
y_k=zeros(m,num_labels);
for k=1:num_labels
    y_k(:,k) = y==k;
end

%cost function
J = 0;
for i=1:m
    for k=1:num_labels
        J=J + -y_k(i,k)*log(a3(i,k))-(1-y_k(i,k))*log(1-a3(i,k));
    end
end
J = J/m;
J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

%gradient of cost function
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
for t = 1:m
   delta_3=(a3(t,:)-y_k(t,:))';
   delta_2=Theta2(:,2:end)'*delta_3.*sigmoidGradient(z2(t,:))';
   Theta1_grad = Theta1_grad + delta_2*a1(t,:);
   Theta2_grad = Theta2_grad + delta_3*a2(t,:);
end
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m)*Theta2(:,2:end);
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
