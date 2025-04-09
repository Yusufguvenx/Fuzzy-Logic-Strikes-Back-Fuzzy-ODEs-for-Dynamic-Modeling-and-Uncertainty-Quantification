function ypred = fismodel_u(x, u, number_mf, number_inputs,number_outputs, mbs, learnable_parameters, output_membership_type)

mini_batch_inputs = permute(x, [2 3 1]);

mini_batch_to_be_used = permute([mini_batch_inputs;u], [3 1 2]);

fuzzifed = matrix_fuzzification_layer(mini_batch_to_be_used, "gaussmf", learnable_parameters, number_mf, number_inputs, mbs);
% %
firestrength = firing_strength_calculation_layer(fuzzifed, "product");
% firestrength = firing_strength_calculation_layer(fuzzifed, "deneme");

normalized = firing_strength_normalization_layer(firestrength);



% normalized = softmax_firing_strength_calculation(mini_batch_inputs,learnable_parameters,number_mf,number_inputs,mbs);



%     ypred = defuzzification_layer(mini_batch_inputs, normalized, learnable_parameters, output_membership_type);
ypred = multioutput_defuzzification_layer(mini_batch_to_be_used, normalized, learnable_parameters,number_outputs, output_membership_type);

%resnet structure

ypred = x + ypred;

end