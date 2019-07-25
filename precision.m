function p = precision(pred, y, label)
%PRECISION Caluculates precision for given model
%   p = PRECISION(theta, X, y) computes the precision for a given model
%   using theta

pp = (pred == label);
tp = (pred == y) & pred == label;

pp_sum = sum(pp);
tp_sum = sum(tp);

p = tp_sum / pp_sum;

% =========================================================================


end
