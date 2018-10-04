function [Y, w] = tfidf( X )
%applies TF-IDF weighting to word count vector matrix.

%get inverse document frequencies
w = idf( X );

%TF * IDF
Y = tf( X ) .* repmat( w, 1, size(X,2) );


function Y = tf( X )
%computes word frequencies

Y = X ./ repmat( sum(X,1), size(X,1), 1 );
Y( isnan(Y) ) = 0;


function I = idf(X)
%computes inverse document frequencies

%count the number of words in each document
nz = sum( ( X > 0 ), 2 );

%compute idf for each document
I = log( size(X,2) ./ (nz(:) + 1) );
