function [X,y] = newsgroups_preprocess(vocab_size)
%tf-idf bag of words per-document representation of top vocabulary items in
%newsgroups
    train_sparse=load('newsgroups_data/train.data');
    test_sparse=load('newsgroups_data/test.data');
    train_labels=load('newsgroups_data/train.label');
    test_labels=load('newsgroups_data/test.label');
    test_sparse(:,1)=test_sparse(:,1)+max(train_sparse(:,1));
    X_sparse=[train_sparse;test_sparse];
    y=[train_labels;test_labels];
    counts=accumarray(X_sparse(:,2),X_sparse(:,3));
    [counts_sorted,index]=sort(counts,'descend');
    X_sparse_filtered=X_sparse(ismember(X_sparse(:,2),index(1:vocab_size)),:);
    ids=sort(unique(X_sparse_filtered(:,2)));
    [yes,ranks] = ismember(X_sparse_filtered(:,2),ids);
    X_sparse_filtered(:,2)=ranks;
    X=full(sparse(X_sparse_filtered(:,1),X_sparse_filtered(:,2),X_sparse_filtered(:,3)));
    X = tfidf(X');
    X = X';
end
