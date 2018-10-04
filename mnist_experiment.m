function [acc_base, acc_finetune, acc_fix_f, acc_fix_gf] = mnist_experiment(input_layer_size,hidden_layer_size,max_iter,lambda,m_S,m_T,labels_S,labels_T)
%single MNIST experiment

%set parameters 
options = optimset('MaxIter', max_iter);
num_labels=1; %binary classification task

%load data
X = loadMNISTImages('mnist_data/train-images.idx3-ubyte')';
y = loadMNISTLabels('mnist_data/train-labels.idx1-ubyte');

%shuffle data
sel = randperm(length(y));
X=X(sel,:);
y=y(sel);

%divide data into source and target
X_S=X(1:m_S,:);
y_S=y(1:m_S);
X_T=X((m_S+1):length(y),:);
y_T=y((m_S+1):length(y));

%create the binary labels for source and target tasks
y_S=ismember(y_S,labels_S);
y_T=ismember(y_T,labels_T);

%divide target task data into train and test (should be disjoint)
X_T_train = X_T(1:m_T,:);
y_T_train = y_T(1:m_T);
X_T_test = X_T((round(length(y_T)/2)+1):round(3*length(y_T)/4),:);
y_T_test = y_T((round(length(y_T)/2)+1):round(3*length(y_T)/4));

%initialise parameters
initial_w = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_w = initial_w/norm(initial_w);
initial_v = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_w(:) ; initial_v(:)];

%training

%source training
[w_S, v_S] = train_nn(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X_S,y_S,lambda,options);

%target training from scratch (BASE)
[w_T, v_T] = train_nn(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X_T_train,y_T_train,lambda,options);

%target training with fine-tuning (FINE-TUNE \hat{f})
[w_T_finetune, v_T_finetune] = train_nn_finetune(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X_T_train,y_T_train,lambda,w_S,options);

%target training top layer of weights only (FIX \hat{f})
Z_T_train = sigmoid([ones(size(X_T_train,1), 1) X_T_train] * w_S');
[v_ST] = oneVsAll(Z_T_train, y_T_train, num_labels, lambda,max_iter);

%prediction and evaluation

%BASE
pred_base = predict(w_T, v_T, X_T_test);
acc_base = mean(double(pred_base == y_T_test)) * 100;

%FINE-TUNE \hat{f}
pred_finetune = predict(w_T_finetune,v_T_finetune, X_T_test);
acc_finetune=mean(double(pred_finetune == y_T_test)) * 100;

%FIX \hat{f}
Z_T_test = sigmoid([ones(size(X_T_test,1), 1) X_T_test] * w_S');
pred_fix_f = predictOneVsAll(v_ST, Z_T_test);
acc_fix_f = mean(double(pred_fix_f == y_T_test)) * 100;

%FIX g_S o \hat{f}
pred_fix_gf = predict(w_S, v_S, X_T_test);
acc_fix_gf = mean(double(pred_fix_gf == y_T_test)) * 100;

end