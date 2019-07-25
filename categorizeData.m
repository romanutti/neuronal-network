function y_bin = categorizeData(y)
%CategorizeData Mapps y values to 0 - 4
%   p = CATEGORIZEDATA(Y) Mapps y values to 0 - 4

y_bin = string(y);

y_bin(y_bin == 'Wohnung') = 1;
y_bin(y_bin == 'Sonstiges') = 2;
y_bin(y_bin == 'Einfamilienhaus') = 3;
y_bin(y_bin == 'Mehrfamilienhaus') = 4;

y_bin = str2num(char(y_bin));

% =========================================================================


end
