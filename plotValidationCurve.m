function plotValidationCurve(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, X_val, y_val)
%PLOTVALIDATIONCURVE Plots validation curve for train and cv set
%       PLOTVALIDATIONCURVE(X_train, y_train, X_val, y_val) Plots
%       validation curve for train and cv set
%

[lambda_vec, error_train, error_val] = ...
    validationCurve(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, X_val, y_val);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('Lambda');
ylabel('Error');

end
