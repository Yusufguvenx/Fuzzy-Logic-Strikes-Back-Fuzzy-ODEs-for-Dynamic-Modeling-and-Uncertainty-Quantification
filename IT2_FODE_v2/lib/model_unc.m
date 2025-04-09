function X = model_unc(t, mini_batch_inputs,  ux, number_mf, number_inputs,number_outputs, mbs, learnable_parameters, output_membership_type, input_mf_type, input_type,type_reduction_method,u)

x0 = mini_batch_inputs;

step = t(end) - t(1);

PP = griddedInterpolant(t, permute(ux,[2, 1, 3]), "previous");
fcn = @(t, x, param) odemodel(t, x, param,  number_mf, number_inputs,number_outputs, mbs, PP, output_membership_type, input_mf_type, input_type, type_reduction_method, u);
X = dlode45(fcn, t, x0, learnable_parameters, DataFormat='SCB', GradientMode = "direct", AbsoluteTolerance=0.01, RelativeTolerance=0.01, MaxStepSize=step, InitialStepSize= step);
X = permute(X, [2, 4, 3, 1]);
end

