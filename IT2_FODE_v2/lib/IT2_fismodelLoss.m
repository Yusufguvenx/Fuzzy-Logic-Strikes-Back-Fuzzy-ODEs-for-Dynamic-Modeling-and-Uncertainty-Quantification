function [loss, gradients, yPred_lower, yPred_upper, yPred] = IT2_fismodelLoss(x, number_inputs, y,number_outputs, number_mf, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u)

[yPred_lower, yPred_upper, yPred] = IT2_fismodel(x, number_mf, number_inputs,number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u);


% loss = log_cosh_loss(yPred, y, 1);

% loss_tilted = tilted_loss(y, yPred_lower, yPred_upper, 0.005, 0.995,1);


loss = log_cosh_loss(yPred, y, mbs);
% loss = 0;



% loss = tilted_loss(y, yPred, yPred, 0.5, 0.5,mbs)/2;

% 
loss_tilted = tilted_loss(y, yPred_lower, yPred_upper, 0.005, 0.995,mbs);
% %% 99 cover
% loss_tilted = tilted_loss(y, yPred_lower, yPred_upper, 0.025, 0.975, mbs);% %% 95 cover
% loss_tilted = tilted_loss(y, yPred_lower, yPred_upper, 0.1, 0.9, mbs);% %% 80 cover


loss = sum((loss + loss_tilted),2);


% loss = sum((loss),2);


gradients = dlgradient(loss, learnable_parameters);

end