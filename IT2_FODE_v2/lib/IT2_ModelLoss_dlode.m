function [loss, gradients, yPred_lower, yPred_upper, yPred] = IT2_ModelLoss_dlode(t, x, number_inputs, ux, y,number_outputs, number_mf, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u, ahead)

[yPred_lower, yPred_upper, yPred] = model(t,x, ux, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u);

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

acc_loss = l1loss(X_temp, y, "NormalizationFactor","batch-size", DataFormat="STB");

% loss_RQR = RQR_loss(diff_ground, yPred_lower, yPred_upper, 0.99, mbs);
loss_RQRW = RQRW_loss(diff_ground, yPred_lower, yPred_upper, 0.99, 0.5, mbs);

% loss_tilted = tilted_loss(diff_ground, yPred_lower, yPred_upper, 0.005, 0.995, mbs);
loss =  loss_RQRW + acc_loss;
% loss = sum((loss + loss_tilted),2);

% loss = loss_tilted;
% loss = sum((loss),2);


gradients = dlgradient(loss, learnable_parameters);

end