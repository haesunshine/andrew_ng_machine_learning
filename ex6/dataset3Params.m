function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
s = size(vals)(1);
result = zeros(s * s, 3);
counter = 1;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
  for i = 1:s
  	for j = 1:s
  		C_pos = vals(i);
  		sigma_pos = vals(j);

  		model = svmTrain(X, y, C_pos, @(x1, x2) gaussianKernel(x1, x2, sigma_pos));
  		predictions = svmPredict(model, Xval);
  		
  		result(counter, 1) = C_pos;
  		result(counter, 2) = sigma_pos;
  		result(counter, 3) = mean(double(predictions ~= yval));
  		counter += 1;

  	end
  end

[min_error_values min_error_indices] = min(result(:, 3));

C = result(min_error_indices, 1);
sigma = result(min_error_indices, 2);






% =========================================================================

end
