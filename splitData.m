function [X_val, y_val, X_train, y_train] = splitData(X, y)
%SplitData Splits data into train and cross validation part
%   [X_val, y_val, X_train, y_train] = SPLITDATA(X, y) Splits data into 
%   train and cross validation part

% Determine how many rows 20% is.
[rows, columns] = size(X);
% Determine the last row number upper 20% of rows.
lastRow = int32(floor(0.2 * rows));
% Get first 20% into a cross validation dataset
X_val = X(1:lastRow, :);
y_val = y(1:lastRow, :);
% Get the rest into a train dataset
X_train = X(lastRow+1:end, :);
y_train = y(lastRow+1:end, :);

% =========================================================================


end
