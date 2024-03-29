function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

%X_norm = X;
%mu = zeros(1, size(X, 2));
%sigma = zeros(1, size(X, 2));

%for col = 1:size(X,2)
%    mu = mean(X(:,col));
%    sigma = std(X(:,col));
%    X_norm(:,col) = X_norm(:,col) - mu;
%    X_norm(:,col) = X_norm(:,col) / sigma;
%end


% ============================================================

end
