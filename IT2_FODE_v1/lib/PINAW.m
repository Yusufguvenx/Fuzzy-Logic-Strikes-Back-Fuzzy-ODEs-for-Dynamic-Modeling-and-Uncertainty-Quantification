function x = PINAW(y, y_l, y_u)
    % Assuming y, y_l, and y_u are matrices with the same size
    % Each row represents a different output
    % Initialize the output array
    n_outputs = size(y, 1); % Number of outputs/rows
    x = zeros(1, n_outputs); % PINAW values for each output

    for i = 1:n_outputs
        % Extract the ith row/output from each matrix
        y_i = y(i, :);
        y_l_i = y_l(i, :);
        y_u_i = y_u(i, :);

        % Number of samples
        n = length(y_l_i);
        
        % Range of the actual values
        r = max(y_i) - min(y_i);

        % Calculate and store the PINAW value for the ith output
        x(i) = sum(y_u_i - y_l_i) / (n * r);
    end
end
