function table = importHouseData(f, n)
%IMPORTHOUSEDATA Imports the house data into a new table 
%   IMPORTHOUSEDATA(file, n) imports the data of a given file f into a table and
%   plots the first n lines.

% Import the file 
formatSpec = '%d%f%f%f%s%d%d%s%f%f%d%d%d%d%d%d%d%d%d';   
table = readtable(f, 'Format', formatSpec, 'FileEncoding', 'UTF-8');

% Plot the first n rows
head(table,n)

% ============================================================

end
