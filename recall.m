function p = recall(pred, y, label)
%RECALL Caluculates recall for given model
%   p = PRECISION(theta, X, y) computes the recall for a given model
%   using theta

m = size(pred, 1); % Number of training examples

pp = (pred == label);
tp = (pp == y) & pp == 1;
ap = (y == label);

tp_sum = sum(tp);
ap_sum = sum(ap);

p = tp_sum / ap_sum;

% =========================================================================


end
