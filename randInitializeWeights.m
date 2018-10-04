function W = randInitializeWeights(L_in, L_out)
%randomly initialize the weights of a layer with L_in
%based on code from Andrew Ng Coursera course on Machine Learning

epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end
