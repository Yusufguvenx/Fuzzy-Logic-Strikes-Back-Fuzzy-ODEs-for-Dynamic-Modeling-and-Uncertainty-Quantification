function [output_lower, output_upper, output_mean] = IT2_fismodel(x, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u)

[fuzzifed_lower, fuzzifed_upper] = T2_matrix_fuzzification_layer(x, input_mf_type,input_type, learnable_parameters, number_mf, number_inputs, mbs);


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
[output_lower, output_upper, output_mean] = T2_multioutput_defuzzification_layer(x, firestrength_lower, firestrength_upper, learnable_parameters,number_outputs, output_type,type_reduction_method,mbs,number_mf,u);
% [output_lower, output_upper, output_mean] = T2_multioutput_defuzzification_layer(x, normalized_firestrength_lower, normalized_firestrength_upper, learnable_parameters,number_outputs, output_type,type_reduction_method,mbs,number_mf,u);







%%
% 


end