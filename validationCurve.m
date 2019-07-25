function [lambda_vec, error_train, error_val] = ...
    validationCurve(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, X_val, y_val)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.0001, 0.0003, 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

   
for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);

    % Train NN
    % Set options
    options = optimset('MaxIter', 50);

    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
                                  input_layer_size, ...
                                  hidden_layer_size, ...
                                  num_labels, X_train, y_train, lambda);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
        
    error_train(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                     num_labels, X_train, y_train, 0);
    error_val(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                     num_labels, X_val, y_val, 0);
    
end

end
