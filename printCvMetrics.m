function p = printCvMetrics(pred, y, cm, a)
%PrintCVMetrics Prints most important metrics
%   p = PRINTCVMETRICS(pred, y, cm, a) outputs the most important metrics

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

disp(['Cross-validation Confusion Matrix:' newline, ...
      '[' num2str(cm(:).') ']' newline, ...
      'Cross-validation Accuracy (%): ', num2str(a) newline, ...
      newline, ... 
      'Cross-validation Precision, Wohnung: ', num2str(p1) newline, ...
      'Cross-validation Recall, Wohnung: ', num2str(r1) newline, ...
      'Cross-validation F score, Wohnung: ', num2str(f1) newline, ...
        newline, ... 
      'Cross-validation Precision, Sonstiges: ', num2str(p2) newline, ...
      'Cross-validation Recall, Sonstiges: ', num2str(r2) newline, ...
      'Cross-validation F score, Sonstiges: ', num2str(f2) newline, ...
        newline, ... 
      'Cross-validation Precision, Einfamilienhaus: ', num2str(p3) newline, ...
      'Cross-validation Recall, Einfamilienhaus: ', num2str(r3) newline, ...
      'Cross-validation F score, Einfamilienhaus: ', num2str(f3) newline, ...
        newline, ... 
      'Cross-validation Precision, Mehrfamilienhaus: ', num2str(p4) newline, ...
      'Cross-validation Recall, Mehrfamilienhaus: ', num2str(r4) newline, ...
      'Cross-validation F score, Mehrfamilienhaus: ', num2str(f4)]);

% =========================================================================


end
