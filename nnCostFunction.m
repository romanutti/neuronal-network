function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

K = num_labels;

% Create Y as k-dimensional matrix
e = eye(K);
Y = e(y,:);

% Part 1: FeedForward
a1 = [ones(m, 1), X];

z2 = a1 * Theta1';
a2 = [ones(size(z2, 1), 1), sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

reg = lambda / (2*m) * (sum(sum(Theta1(:, 2:end) .^2)) + sum(sum(Theta2(:, 2:end) .^2)));
cost = sum((-Y .* log(a3)) - (1 - Y) .* log(1 - a3));
J = (1 / m) * sum(cost) + reg;

delta1cum = 0;
delta2cum = 0;

% Part 2: Backpropagation
for t = 1:m
    % Step 1
    a1 = [1; X(t, :)']; % Bias

    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)]; % Bias

    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    % Step 2
    delta3 = a3 - Y(t,:)';
    
    % Step 3
    delta2 = (Theta2(:, 2:end)'... // remove bias
              * delta3) .* sigmoidGradient(z2);
          
    % Step 4
    delta2cum = delta2cum +(delta3 * a2');
    delta1cum = delta1cum +(delta2 * a1');
end

% Step 5
Theta1_grad = (1/m) * delta1cum;
Theta2_grad = (1/m) * delta2cum;

% Regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
