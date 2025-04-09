function plotPredictions(ground_truths, preds)
    % Determine the size of your data
    [num_outputs, num_samples] = size(ground_truths);
    
    % Create a figure to hold all subplots
    fig = figure;
    
    % Define the position for the figure window
    position = [100, 100, 1200, 800]; % [left, bottom, width, height]
    
    % Set the position of the figure window
    set(fig, 'Position', position);
    
    for i = 1:num_outputs
        % Create a subplot for each output
        subplot(num_outputs, 1, i);
        hold on; % Allows multiple plots in the same subplot
        
        % Plot ground truth
        plot(1:num_samples, ground_truths(i, :), 'rx', 'LineWidth', 1); % Ground truth with red x markers
        
        % Plot predictions
        plot(1:num_samples, preds(i, :), 'bo', 'LineWidth', 1); % Predictions with blue circle markers
        

        
        % Customize the plot
        legend('Ground Truth', 'Predictions', 'Location', 'Best');
        xlabel('Sample');
        ylabel(sprintf('Output %d', i));
        title(sprintf('Output %d Predictions vs. Ground Truth with Bounds', i));
        hold off; % Finished plotting on this subplot
    end
    
    % Adjust subplot spacing
    sgtitle('All Outputs: Predictions, Ground Truths');

end
