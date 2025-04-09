function [output_lower, output_upper, output_mean] = GT2_fismodel_LA0(x, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u,alpha, delta)

%[fuzzified_lower, fuzzified_upper] = T2_fuzzification_layer(x, input_mf_type,input_type, learnable_parameters, number_mf, number_inputs, mbs);
[fuzzified_lower, fuzzified_upper] = T2_matrix_fuzzification_layer(x, input_mf_type,input_type, learnable_parameters, number_mf, number_inputs, mbs);


for i=1:numel(alpha)


    cut(i).smf_fuzzified_lower = dlarray(0);
    cut(i).smf_fuzzified_upper = dlarray(0);
    cut(i).firestrength_lower = dlarray(0);
    cut(i).firestrength_upper = dlarray(0);
    cut(i).output_lower = dlarray(0);
    cut(i).output_upper = dlarray(0);
    cut(i).output_mean = dlarray(0);
    
end

% parfor (i=1:numel(alpha), 'debug')
for i=1:numel(alpha)

    [cut(i).smf_fuzzified_lower,cut(i).smf_fuzzified_upper] = GT2_matrix_fuzzification_layer(fuzzified_lower,fuzzified_upper, alpha(i), learnable_parameters, delta);
    [cut(i).firestrength_lower, cut(i).firestrength_upper] = T2_firing_strength_calculation_layer(cut(i).smf_fuzzified_lower, cut(i).smf_fuzzified_upper, "product");
    [cut(i).output_lower, cut(i).output_upper, cut(i).output_mean] = T2_defuzzification_layer(x, cut(i).firestrength_lower, cut(i).firestrength_upper, learnable_parameters,number_outputs, output_type,type_reduction_method,mbs,number_mf,u);

end

output_lower = dlarray(0);
output_upper = dlarray(0);
output_mean = dlarray(0);
% 
% parfor (i=1:numel(alpha), 'debug')
for i=1:numel(alpha)

    output_lower = output_lower + alpha(i).* cut(i).output_lower;
    output_upper = output_upper + alpha(i).* cut(i).output_upper;
    output_mean = output_mean + alpha(i).* cut(i).output_mean;

end


output_lower = output_lower / sum(alpha); %y_lower
output_upper = output_upper / sum(alpha); %y_upper
output_mean = output_mean / sum(alpha); %y_star