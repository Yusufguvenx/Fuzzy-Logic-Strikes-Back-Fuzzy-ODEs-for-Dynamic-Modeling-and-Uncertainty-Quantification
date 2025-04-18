function [loss, gradients, yPred_lower, yPred_upper, yPred] = GT2_fismodelLoss(x, number_inputs, y,number_outputs, number_mf, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u, alpha, delta, alpha_rev)

[yPred_lower, yPred_upper, yPred] = GT2_fismodel_LA1_new(x, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha, delta, alpha_rev);

% yPred = (yPred_lower(3, :, :) + yPred_upper(3, :, :)) / 2;

% yPred_2 = (yPred_upper2 + yPred_lower2) / 2;

% yPred = sigmoid(yPred);


loss = log_cosh_loss(yPred, y, mbs);
% loss = crossentropy(yPred, y, "NormalizationFactor","batch-size", "DataFormat","SCB");

% loss_mse = mse(yPred, y, "DataFormat","SCB");

% loss_tilted = tilted_loss(y, yPred_lower, yPred_upper, 0.025, 0.975, mbs);
loss_tilted1 = tilted_loss(y, yPred_lower, yPred_upper, 0.005, 0.995, mbs); %second alphacut
% loss_tilted2 = tilted_loss(y, yPred_lower(2, :, :), yPred_upper(2, :, :), 0.5, 0.5, mbs) / 2; %last alphacut
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

loss = loss + loss_tilted1;

gradients = dlgradient(loss, learnable_parameters);


end