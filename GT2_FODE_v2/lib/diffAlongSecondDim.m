function diffVectors = diffAlongSecondDim(vectors)
    % Check if input is a 3D array
    if ndims(vectors) ~= 3
        error('Input must be a 3D array');
    end
    
    % Calculate the difference along the second dimension
    diffVectors = vectors(:, 2:end, :) - vectors(:, 1:end-1, :);
end
