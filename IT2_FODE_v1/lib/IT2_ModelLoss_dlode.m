function [loss, gradients, yPred_lower, yPred_upper, yPred] = IT2_ModelLoss_dlode(t, x, number_inputs, ux, y,number_outputs, number_mf, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u, ahead)

[yPred_lower, yPred_upper, yPred] = model(t,x, ux, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u);



% acc_loss = log_cosh_loss(yPred, y, mbs, 1);
acc_loss = l1loss(yPred, y, "NormalizationFactor","batch-size", DataFormat="STB");

loss_RQR = RQR_loss(y, yPred_lower, yPred_upper, 0.99, mbs);
% loss_RQRW = RQRW_loss(y, yPred_lower, yPred_upper, 0.99, 0.5, mbs);
% loss_tilted = tilted_loss(y, yPred_lower, yPred_upper, 0.005, 0.995, mbs, 1);
loss =  acc_loss + loss_RQR;
% loss = acc_loss;
% loss = sum((loss + loss_tilted),2);

% loss = loss_tilted;
% loss = sum((loss),2);


gradients = dlgradient(loss, learnable_parameters);

end