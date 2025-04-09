function plotPredictions(ground_truths, preds, lower_bounds, upper_bounds)
    % Determine the size of your data
    [num_outputs, num_samples] = size(ground_truths);
    
    % Create a figure to hold all subplots
    figure;
    
    for i = 1:num_outputs
        % Create a subplot for each output
        subplot(num_outputs, 1, i);
        hold on; % Allows multiple plots in the same subplot
        
        % Plot ground truth
        plot(1:num_samples, ground_truths(i, :), 'rx', 'LineWidth', 1); % Ground truth with red x markers
        
        % Plot predictions
        plot(1:num_samples, preds(i, :), 'bo', 'LineWidth', 1); % Predictions with blue circle markers
        
        % Plot lower and upper bounds
        plot(1:num_samples, lower_bounds(i, :), 'g-', 'LineWidth', 1); % Lower bounds in green
        plot(1:num_samples, upper_bounds(i, :), 'm-', 'LineWidth', 1); % Upper bounds in magenta
        
        % Customize the plot
        legend('Ground Truth', 'Predictions', 'Lower Bound', 'Upper Bound', 'Location', 'Best');
        xlabel('Sample');
        ylabel(sprintf('Output %d', i));
        title(sprintf('Output %d Predictions vs. Ground Truth with Bounds', i));
        hold off; % Finished plotting on this subplot
    end
    
    % Adjust subplot spacing
    sgtitle('All Outputs: Predictions, Ground Truths, Lower and Upper Bounds');
    set(gcf, 'Position', get(0, 'Screensize')); % Optional: Maximize figure window
end
