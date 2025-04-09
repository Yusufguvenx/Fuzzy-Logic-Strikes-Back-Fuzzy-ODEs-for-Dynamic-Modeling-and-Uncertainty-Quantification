function [output_lower, output_upper, output_mean] = evaluatemodel(x, u_mini_batch, learnable_parameters, number_mf, number_inputs, number_outputs, mbs, output_type, input_mf_type, input_type, type_reduction_method, u, alpha, delta, alpha_rev)

mini_batch_inputs = permute(x, [2 3 1]);

mini_batch_to_be_used = permute([mini_batch_inputs;u_mini_batch], [3 1 2]);


fuzzified = T2_matrix_fuzzification_layer(mini_batch_to_be_used, input_mf_type,input_type, learnable_parameters, number_mf, number_inputs, mbs);

[smf_fuzzified_lower,smf_fuzzified_upper] = GT2_matrix_fuzzification_layer(fuzzified, alpha, learnable_parameters, delta);
[firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(smf_fuzzified_lower, smf_fuzzified_upper, "product");
firestrength_lower = permute(firestrength_lower, [1, 4, 3, 2]);
firestrength_upper = permute(firestrength_upper, [1, 4, 3, 2]);
[output_lower, output_upper, output_mean] = GT2_defuzzification_layer(mini_batch_to_be_used, firestrength_lower, firestrength_upper, learnable_parameters,number_outputs, output_type,type_reduction_method,mbs,number_mf,u);

output_lower = output_lower(1, :, :, :); %added
output_upper = output_upper(1, :, :, :); %added

output_mean = pagemtimes(alpha_rev, output_mean);
% output_mean = output_mean(3, :, :, :);%added
output_mean = permute(output_mean, [1 4 3 2]); %added
output_lower = permute(output_lower, [1 4 3 2]);%added
output_upper = permute(output_upper, [1 4 3 2]);%added

output_mean = x + output_mean;%next state
output_lower = x + output_lower;%next state lower
output_upper = x + output_upper;%next state upper


% output_mean = sigmoid(output_mean);

end


