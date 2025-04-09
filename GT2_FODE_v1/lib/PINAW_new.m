function x = PINAW_new(y, y_l, y_u)

 % Assuming y, y_l, and y_u are matrices with the same size
    % Each row represents a different output
    % Initialize the output array
    n_outputs = size(y, 1); % Number of outputs/rows
    % x = zeros(1, n_outputs); % PICP values for each output

    % for i = 1:n_outputs
    %     % Extract the ith row/output from each matrix
    %     y_i = y(i, :);
    %     y_l_i = y_l(i, :);
    %     y_u_i = y_u(i, :);
    % 
    %     % Calculate indicators for values outside the prediction interval
    %     i_u = y_i > y_u_i;
    %     i_l = y_i < y_l_i;
    % 
    %     % Count the number of values outside the prediction interval
    %     n_u = sum(i_u);
    %     n_l = sum(i_l);
    % 
    %     % Total number of samples
    %     n = length(y_i);
    % 
    %     % Calculate the number of samples within the prediction interval
    %     j = n - (n_u + n_l);
    % 
    %     % Calculate and store the PICP value for the ith output
    %     x(i) = 100 / n * j;
    % end

    n = size(y, 2);


    diff_ = (y_u - y_l);
    area_per_sample = prod(diff_, 1);
    area = sum(area_per_sample);
    ranges = (max(y, [], 2) - min(y, [], 2));
    multiplied_ranges = prod(ranges);
    normalized_area = area/multiplied_ranges;
    normalized_area = normalized_area / n;


    x = normalized_area;



end

