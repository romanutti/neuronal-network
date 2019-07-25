function [Theta1, Theta2, Theta3] = reshapeThetaNNN(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels)
%ReshapeThetaNNN unrolls thetas
%   [J grad] = RESHAPETHETANNN(nn_params, hidden_layer_size, num_labels) unrolls thetas.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 3 layer neural network
prevTo = hidden_layer_size * (input_layer_size + 1);
% Theta1
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

%Theta3
Theta3 = reshape(data, ...
                 num_labels, (hidden_layer_size + 1));

end
