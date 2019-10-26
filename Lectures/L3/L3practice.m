function predictions = predict(W1,W2, X)

% Your code goes here.

% hidden layer (logistic)
z = W1' * X';
y1 = 1 ./ (1 + e.^(-z));

% output layer (linear)
y = W2' * y1;
predictions = y'
endfunction