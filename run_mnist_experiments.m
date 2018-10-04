%MNIST experiments

%set parameters
rng(4980);
input_layer_size=784;
hidden_layer_size=50; 
max_iter=200;
lambda=1;
m_S=50000;
m_T=500;
num_result_columns=4;
num_labels=10;
overlaps=(0:(num_labels/2)); %number of labels overlapping in positive class of source and target tasks
results=zeros(length(overlaps),num_result_columns);

num_trials=10;
for trial = 1:num_trials
    labels_random=randperm(num_labels);
    
    %construct the source task positive class by randomly selecting half the labels
    labels_S=labels_random(1:(num_labels/2));
    labels_remaining=labels_random((num_labels/2+1):num_labels);
    trial
    
    %only need to consider cases where overlap is at least half of the
    %source task labels (otherwise just swap the target classes)
    for i = round(length(overlaps)/2+1):length(overlaps)
        overlap=overlaps(i);
        overlap
        
        %construct the target task positive class by randomly selecting from the source task positive class 
        %labels up to the overlap amount, 
        %plus randomly selecting from the other labels to make up half the labels in total
        labels_T=[labels_S(1:overlap),labels_remaining(1:(num_labels/2-overlap))];
        
        %accuracy for [BASE, FINE-TUNE \hat{f}, FIX \hat{f}, FIX g_S o \hat{f}]
        [acc_base, acc_finetune, acc_fix_f, acc_fix_gf]=mnist_experiment(input_layer_size,hidden_layer_size,max_iter,lambda,m_S,m_T,labels_S,labels_T);    
        
        % average results over all trials
        results(i,:)=((trial-1)*results(i,:)+[acc_base, acc_finetune, acc_fix_f, acc_fix_gf])/trial;
        
    end
end