function [output_lower, output_upper, output_mean] = odemodel(x, u_mini_batch, learnable_parameters, number_mf, number_inputs,number_outputs, mbs, output_type, input_mf_type, input_type, type_reduction_method, u)

mini_batch_inputs = permute(x, [2 3 1]);
mini_batch_to_be_used = permute([mini_batch_inputs;u_mini_batch], [3 1 2]);


[fuzzifed_lower, fuzzifed_upper] = T2_matrix_fuzzification_layer(mini_batch_to_be_used, input_mf_type,input_type, learnable_parameters, number_mf, number_inputs, mbs);


% fuzzifed_lower(fuzzifed_lower<=10^-2)=0;
% fuzzifed_upper(fuzzifed_upper<=10^-2)=0;

if input_type == "H&softmax" || input_type == "S&softmax" || input_type == "HS&softmax"
    [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper, "product");
else
    % [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper, "product");
    % [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper, "min");
    % [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper, "bounded-product");
    % [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper, "mean");
    [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper, "deneme");
    % [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper, "deneme2");
    % [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper, "deneme3");
end
%layer norm v1

% [firestrength_lower,m,s] = zscore_norm(firestrength_lower);
% firestrength_lower = sigmoid(firestrength_lower);
% firestrength_lower = (firestrength_lower.*s)+m;
% 
% firestrength_upper = (firestrength_upper -m)./(s+(1e-32));
% firestrength_upper = sigmoid(firestrength_upper);
% firestrength_upper = (firestrength_upper.*s)+m;





% NORM
% [normalized_firestrength_lower,normalized_firestrength_upper] = IT2_firing_strength_normalization_layer(firestrength_lower,firestrength_upper);
% 
[output_lower, output_upper, output_mean] = T2_multioutput_defuzzification_layer(mini_batch_to_be_used, firestrength_lower, firestrength_upper, learnable_parameters,number_outputs, output_type,type_reduction_method,mbs,number_mf,u);

output_lower = x + output_lower;%x_l(k+1)
output_upper = x + output_upper;%x_u(k+1)
output_mean = x + output_mean;%x_m(k+1)

end
