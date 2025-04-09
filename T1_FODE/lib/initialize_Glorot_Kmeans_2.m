function [learnable_parameters] = initialize_Glorot_Kmeans_2(number_inputs, number_outputs, number_mf, output_type)


centers = rand(number_mf, number_inputs)*0.01;

learnable_parameters.input_centers = centers;

learnable_parameters.input_centers = dlarray(learnable_parameters.input_centers);

%% sigmas

s = rand(1, number_inputs)*0.01; 
s(s == 0) = 1;


s = repmat(s,number_mf,1);
learnable_parameters.input_sigmas = s;

learnable_parameters.input_centers = dlarray(learnable_parameters.input_centers);
learnable_parameters.input_sigmas = dlarray(learnable_parameters.input_sigmas);

if output_type == "singleton"

    c = rand(number_mf,number_outputs)*0.01;
    learnable_parameters.singleton.c = dlarray(c);

elseif output_type == "linear"

    a = rand(number_mf*number_outputs,number_inputs)*0.01;
    learnable_parameters.linear.a = dlarray(a);

    b = rand(number_mf*number_outputs,1)*0.01; % single output
    learnable_parameters.linear.b = dlarray(b);

elseif output_type == "IV"

    c = rand(number_mf,number_outputs)*0.01;
    delta = rand(number_mf,number_outputs)*0.01;

    learnable_parameters.IV.c = dlarray(c);
    learnable_parameters.IV.delta = dlarray(delta);

elseif output_type == "IVL"

    a = rand(number_mf*number_outputs,number_inputs)*0.01;
    b = rand(number_mf*number_outputs,1)*0.01; % single output

    delta_a = rand(number_mf*number_outputs,number_inputs)*0.01;
    delta_b = rand(number_mf*number_outputs,1)*0.01; % single output

    learnable_parameters.IVL.a = dlarray(a);
    learnable_parameters.IVL.delta_a = dlarray(delta_a);
    learnable_parameters.IVL.b = dlarray(b);
    learnable_parameters.IVL.delta_b = dlarray(delta_b);

end

end