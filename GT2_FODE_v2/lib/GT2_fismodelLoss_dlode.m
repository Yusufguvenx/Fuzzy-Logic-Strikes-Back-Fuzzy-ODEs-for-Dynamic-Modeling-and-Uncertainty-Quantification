function [loss, gradients, yPred_lower, yPred_upper, yPred] = GT2_fismodelLoss_dlode(t, x, number_inputs, ux, y, number_outputs, number_mf, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u, alpha, delta, alpha_rev, ahead)

[yPred_lower, yPred_upper, yPred] = model(t, x, ux, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha, delta, alpha_rev);

x0 = permute(x, [2 1 3]);
x = x0;
X_temp = [];
for ct = 1:ahead
    y_ = yPred(:, ct, :) + x;
    X_temp = [X_temp y_];
    x = y_;
end

y_con = cat(2, x0, y);
diff_ground = diffAlongSecondDim(y_con);


% loss_RQR = RQR_loss(diff_ground, yPred_lower, yPred_upper, 0.99, mbs);
loss_RQRW = RQRW_loss(diff_ground, yPred_lower, yPred_upper, 0.99, 0.5, mbs);


% acc_loss = log_cosh_loss(X_temp, y, mbs);
acc_loss = l1loss(X_temp, y, "NormalizationFactor","batch-size", DataFormat="STB");
% loss_tilted = tilted_loss(diff_ground, yPred_lower, yPred_upper, 0.005, 0.995,mbs);


loss =  loss_RQRW + acc_loss;

gradients = dlgradient(loss, learnable_parameters);


end