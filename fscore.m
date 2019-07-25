function f = fscore(pred, y, label)
%FSCORE Caluculates fscore for a given label
%   f = FSCORE(pred, y, label) computes the f score for a given label

p = precision(pred, y, label);
r = recall(pred, y, label);

f = 2 * ((p * r) / (p + r));

% =========================================================================


end
