function X = model(t, mini_batch_inputs,  ux, number_mf, number_inputs,number_outputs, mbs, learnable_parameters, output_membership_type)

% x0 = mini_batch_inputs(:, 1:end-1, :);
x0 = mini_batch_inputs;

step = t(end) - t(1);

PP = griddedInterpolant(t, permute(ux,[2, 1, 3]), "pchip");
fcn = @(t, x, param) odeModel(t, x, param,  number_mf, number_inputs,number_outputs, mbs, PP, output_membership_type);
X = dlode45(fcn, t, x0, learnable_parameters, DataFormat='SCB', GradientMode = "direct", AbsoluteTolerance=0.001, RelativeTolerance=0.001, MaxStepSize=step, InitialStepSize= step);
X = permute(X, [2, 4, 3, 1]);
end

