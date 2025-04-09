function [loss, gradients, yPred_lower, yPred_upper, yPred] = GT2_fismodelLoss_dlode(t, x, number_inputs, ux, y, number_outputs, number_mf, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u, alpha, delta, alpha_rev)

[yPred_lower, yPred_upper, yPred] = model(t, x, ux, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha, delta, alpha_rev);


% loss = log_cosh_loss(yPred, y, mbs);
loss_x = l1loss(yPred, y, "NormalizationFactor","batch-size", DataFormat="STB");

% loss_one_step = l2loss(yPred(:, 2, :), y(:, 2, :), "NormalizationFactor","batch-size", DataFormat="STB");


% dx = y(:, 2:end, :) - y(:, 1:end-1, :);
% dx_pred = yPred(:, 2:end, :) - yPred(:, 1:end-1, :);
% 
% loss_dx = l2loss(dx_pred, dx, "NormalizationFactor","batch-size", DataFormat="STB");

% loss = crossentropy(yPred, y, "NormalizationFactor","batch-size", "DataFormat","SCB");

% loss_mse = mse(yPred, y, "DataFormat","SCB");
% MPIW_pen = (1/mbs)*sum(abs(yPred_upper - y) + abs(y-yPred_lower), "all");
% loss_tilted = tilted_loss(y, yPred_lower, yPred_upper, 0.005, 0.995, mbs);


% k = abs((yPred_upper + yPred_lower - 2*y)./(yPred_upper - yPred_lower));
% delta = 0.03;



% loss_RQR = RQR_loss(y, yPred_lower, yPred_upper, 0.99, mbs);
loss_RQRW = RQRW_loss(y, yPred_lower, yPred_upper, 0.99, 0.5, mbs);




% % loss_tilted = tilted_loss(y, yPred_lower, yPred_upper, 0.1, 0.9, mbs);
% loss = loss + loss_tilted;

% loss_tilted = loss_tilted1 + loss_tilted2;
% loss_pearce= pearce_loss_redifine(yPred, y, 0.01,  1., alpha);
% loss = loss_pearce + loss;
% loss = loss + loss_tilted1 + loss_tilted2;

% loss_tilted1 = tilted_loss(y, yPred, 0, 1, mbs, alpha);
% loss_tilted2 = logcosh_tilted(y, yPred, alpha, mbs);
% loss_tilted = mse(yPred, y, "DataFormat","SCB");


% loss = 5*loss_tilted1 + loss_tilted2;

loss = loss_x + loss_RQRW;

gradients = dlgradient(loss, learnable_parameters);


end