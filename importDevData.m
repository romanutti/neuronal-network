function table = importDevData(f, n)
%IMPORTDEVDATA Imports the dev data into a new table 
%   IMPORTDEVDATA(file, n) imports the data of a given file f into a table and
%   plots the first n lines.

% Import the file 
formatSpec = '%d%f%f%f';   
table = readtable(f, 'Format', formatSpec, 'FileEncoding', 'UTF-8');

% Plot the first n rows
head(table,n)

% ============================================================

end
