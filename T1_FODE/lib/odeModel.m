function ypred = odeModel(t, mini_batch_inputs, learnable_parameters,  number_mf, number_inputs,number_outputs, mbs, PP, output_membership_type)

if isdlarray(t)
   t = extractdata(t);
end

u = permute(PP(t), [2 3 1]);

mini_batch_inputs = permute(mini_batch_inputs, [2 3 1]);


mini_batch_to_be_used = permute([mini_batch_inputs;u], [3 1 2]);


% fuzzifed = matrix_fuzzification_layer(mini_batch_to_be_used, "gaussmf", learnable_parameters, number_mf, number_inputs, mbs);
% %
% firestrength = firing_strength_calculation_layer(fuzzifed, "product");
% firestrength = firing_strength_calculation_layer(fuzzifed, "deneme");
% 
% normalized = firing_strength_normalization_layer(firestrength);

normalized = softmax_firing_strength_calculation(mini_batch_to_be_used,learnable_parameters,number_mf,number_inputs,mbs);

%     ypred = defuzzification_layer(mini_batch_inputs, normalized, learnable_parameters, output_membership_type);
ypred = multioutput_defuzzification_layer(mini_batch_to_be_used, normalized, learnable_parameters,number_outputs, output_membership_type);


end

