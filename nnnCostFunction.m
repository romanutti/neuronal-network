function [J grad] = nnnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNNCOSTFUNCTION Implements the neural network cost function for a three layer
%neural network which performs classification
%   [J grad] = NNNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 3 layer neural network
prevTo = hidden_layer_size * (input_layer_size + 1);
Theta1 = reshape(nn_params(1:prevTo), hidden_layer_size, (input_layer_size + 1));

% Theta 2
from = prevTo + 1;
to = from - 1 + (hidden_layer_size * (hidden_layer_size + 1));
data = nn_params(from:to);
Theta2 = reshape(data, ...
             hidden_layer_size, (hidden_layer_size + 1));
prevTo = to;
from = prevTo + 1;
data = nn_params(from:end);

Theta3 = reshape(data, ...
                 num_labels, (hidden_layer_size + 1));
                            
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta2));

K = num_labels;

% Create Y as k-dimensional matrix
e = eye(K);
Y = e(y,:);

% Part 1: FeedForward
a1 = [ones(m, 1), X];

z2a = a1 * Theta1';
a2a = [ones(size(z2a, 1), 1), sigmoid(z2a)];

z2b = a2a * Theta2';
a2b = [ones(size(z2b, 1), 1), sigmoid(z2b)];

z3 = a2b * Theta3';
a3 = sigmoid(z3);

reg = lambda / (2*m) * (sum(sum(Theta1(:, 2:end) .^2)) + sum(sum(Theta2(:, 2:end) .^2)));
cost = sum((-Y .* log(a3)) - (1 - Y) .* log(1 - a3));
J = (1 / m) * sum(cost) + reg;

delta1cum = 0;
delta2acum = 0;
delta2bcum = 0;

% Part 2: Backpropagation
for t = 1:m
    % Step 1
    a1 = [1; X(t, :)']; % Bias

    z2a = Theta1 * a1;
    a2a = [1; sigmoid(z2a)]; % Bias
    
    z2b = Theta2 * a2a;
    a2b = [1; sigmoid(z2b)]; % Bias

    z3 = Theta3 * a2b;
    a3 = sigmoid(z3);
    
    % Step 2
    delta3 = a3 - Y(t,:)';
    
    % Step 3b
    delta2b = (Theta3(:, 2:end)'... // remove bias
              * delta3) .* sigmoidGradient(z2b);
    
    % Step 3a
    delta2a = (Theta2(:, 2:end)'... // remove bias
              * delta2b) .* sigmoidGradient(z2a);
          
    % Step 4
    delta2bcum = delta2bcum +(delta3 * a2b');
    delta2acum = delta2acum +(delta2b * a2a');
    delta1cum = delta1cum +(delta2a * a1');
end

% Step 5
Theta1_grad = (1/m) * delta1cum;
Theta2_grad = (1/m) * delta2acum;
Theta3_grad = (1/m) * delta2bcum;

% Regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);
Theta3_grad(:, 2:end) = Theta3_grad(:, 2:end) + (lambda/m) * Theta3(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];

end
