function a = overallAccuracy(pred, y)
%OverallAccuracy Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta and calculates accuracy
%   a = OVERALLACCURACY(theta, X, y) computes accuracy of the predictions for X 
%   using athreshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

p = pred;

a = mean(double(p == y)) * 100;

% =========================================================================

end
