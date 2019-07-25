function p = printTestMetrics(pred, y, cm, a)
%PrintTestMetrics Prints the most important metrics
%   p = PRINTTESTMETRICS(pred, y, cm, a) outputs the most important
%   metrics.

label = 1; % Wohhnung
% Compute precision on our train set
p1 = precision(pred, y, label);
% Compute recall on our train set
r1 = recall(pred, y, label);
% Compute f score on our train set
f1 = fscore(pred, y, label);

label = 2; % Sonstiges
% Compute precision on our train set
p2 = precision(pred, y, label);
% Compute recall on our train set
r2 = recall(pred, y, label);
% Compute f score on our train set
f2 = fscore(pred, y, label);

label = 3; % Einfamilienhaus
% Compute precision on our train set
p3 = precision(pred, y, label);
% Compute recall on our train set
r3 = recall(pred, y, label);
% Compute f score on our train set
f3 = fscore(pred, y, label);

label = 4; % Mehrfamilienhaus
% Compute precision on our train set
p4 = precision(pred, y, label);
% Compute recall on our train set
r4 = recall(pred, y, label);
% Compute f score on our train set
f4 = fscore(pred, y, label);

disp(['Test Confusion Matrix:' newline, ...
      '[' num2str(cm(:).') ']' newline, ...
      'Test Accuracy (%): ', num2str(a) newline, ...
      newline, ... 
      'Test Precision, Wohnung: ', num2str(p1) newline, ...
      'Test Recall, Wohnung: ', num2str(r1) newline, ...
      'Test F score, Wohnung: ', num2str(f1) newline, ...
        newline, ... 
      'Test Precision, Sonstiges: ', num2str(p2) newline, ...
      'Test Recall, Sonstiges: ', num2str(r2) newline, ...
      'Test F score, Sonstiges: ', num2str(f2) newline, ...
        newline, ... 
      'Test Precision, Einfamilienhaus: ', num2str(p3) newline, ...
      'Test Recall, Einfamilienhaus: ', num2str(r3) newline, ...
      'Test F score, Einfamilienhaus: ', num2str(f3) newline, ...
        newline, ... 
      'Test Precision, Mehrfamilienhaus: ', num2str(p4) newline, ...
      'Test Recall, Mehrfamilienhaus: ', num2str(r4) newline, ...
      'Test F score, Mehrfamilienhaus: ', num2str(f4)]);

% =========================================================================


end
