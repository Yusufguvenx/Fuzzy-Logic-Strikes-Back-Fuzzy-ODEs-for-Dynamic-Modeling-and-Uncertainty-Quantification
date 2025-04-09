function [learnable_parameters, timesteps] = initialize_Glorot_IT2_dlode(input_data, number_inputs, number_outputs, input_type ,output_type, number_mf,type_reduction_method, t, neuralOdeTimeSteps)


%% centers with Kmean

dt = t(2);
timesteps = (0:neuralOdeTimeSteps)*dt;

% data = [input_data output_data];
data = input_data;
data = extractdata(permute(data,[3 2 1]));

for i=1:number_inputs
[~,centers(:,i)] = kmeans(data(:,i),number_mf);
end

learnable_parameters.input_centers = centers;

learnable_parameters.input_centers = dlarray(learnable_parameters.input_centers);

%% sigmas

s = std(data); 
s(s == 0) = 1;
s = repmat(s,number_mf,1);

learnable_parameters.input_sigmas = s;
learnable_parameters.input_sigmas = dlarray(learnable_parameters.input_sigmas);

if input_type == "H" || input_type == "Hv2" || input_type == "H&softmax"

    h = rand(number_mf,number_inputs);

    learnable_parameters.input_h = h;

    learnable_parameters.input_h = dlarray(learnable_parameters.input_h);

elseif input_type == "S" || input_type == "S&softmax"

    delta_sigma = rand(number_mf,number_inputs)*0.01;

    learnable_parameters.delta_sigmas = delta_sigma;

    learnable_parameters.delta_sigmas = dlarray(learnable_parameters.delta_sigmas);

elseif input_type == "HS" || input_type == "Hv2S" || input_type == "HS&softmax"

    h = rand(number_mf,number_inputs);

    learnable_parameters.input_h = h;

    learnable_parameters.input_h = dlarray(learnable_parameters.input_h);

    delta_sigmas = rand(number_mf,number_inputs)*0.01;

    learnable_parameters.delta_sigmas = delta_sigmas;

    learnable_parameters.delta_sigmas = dlarray(learnable_parameters.delta_sigmas);

end

%%

if output_type == "singleton"

    c = rand(number_mf,number_outputs)*0.01;
    learnable_parameters.singleton.c = dlarray(c);

    if type_reduction_method == "BMM" || type_reduction_method == "NT_alpha" || type_reduction_method == "NT_alpha2" || type_reduction_method == "KM_BMM" || type_reduction_method == "CQTR"
        alpha = (rand(1,number_outputs)*2)-1;
        learnable_parameters.singleton.alpha = dlarray(alpha);

    elseif type_reduction_method == "NT_multi_alpha"  || type_reduction_method == "CQTR_multi_alpha"
        alpha = (rand(number_mf,number_outputs)*2)-1;
        learnable_parameters.singleton.alpha = dlarray(alpha);

    elseif type_reduction_method == "NTv2" || type_reduction_method == "NTv2_alpha2" || type_reduction_method == "NTv3" || type_reduction_method == "NTv3_alpha2"
        alpha = (rand(1,number_outputs)*2)-1;
        beta = (rand(1,number_outputs)*2)-1;
        learnable_parameters.singleton.alpha = dlarray(alpha);
        learnable_parameters.singleton.beta = dlarray(beta);

    elseif type_reduction_method == "NTv2_multi_alpha" || type_reduction_method == "NTv3_multi_alpha"  
        alpha = (rand(number_mf,number_outputs)*2)-1;
        beta = (rand(number_mf,number_outputs)*2)-1;
        learnable_parameters.singleton.alpha = dlarray(alpha);
        learnable_parameters.singleton.beta = dlarray(beta);

    elseif type_reduction_method == "NTv4"
        alpha = (rand(number_mf,number_outputs)*1)-0.5;
        beta = (rand(number_mf,number_outputs)*1)-0.5;
        learnable_parameters.singleton.alpha = dlarray(alpha);
        learnable_parameters.singleton.beta = dlarray(beta);
    end

elseif output_type == "linear"

    a = rand(number_mf*number_outputs,number_inputs)*0.01;
    learnable_parameters.linear.a = dlarray(a);


    b = rand(number_mf*number_outputs,1)*0.01; % single output
    learnable_parameters.linear.b = dlarray(b);

    if type_reduction_method == "BMM" || type_reduction_method == "NT_alpha" || type_reduction_method == "NT_alpha2" || type_reduction_method == "KM_BMM" || type_reduction_method == "CQTR"
        alpha = (rand(1,number_outputs)*2)-1;
        learnable_parameters.linear.alpha = dlarray(alpha);

    elseif type_reduction_method == "NT_multi_alpha"  || type_reduction_method == "CQTR_multi_alpha"
        alpha = (rand(number_mf,number_outputs)*2)-1;
        learnable_parameters.linear.alpha = dlarray(alpha);

    elseif type_reduction_method == "NTv2" || type_reduction_method == "NTv2_alpha2" || type_reduction_method == "NTv3" || type_reduction_method == "NTv3_alpha2"
        alpha = (rand(1,number_outputs)*2)-1;
        beta = (rand(1,number_outputs)*2)-1;
        learnable_parameters.linear.alpha = dlarray(alpha);
        learnable_parameters.linear.beta = dlarray(beta);

    elseif type_reduction_method == "NTv2_multi_alpha" || type_reduction_method == "NTv3_multi_alpha"
        alpha = (rand(number_mf,number_outputs)*2)-1;
        beta = (rand(number_mf,number_outputs)*2)-1;
        learnable_parameters.linear.alpha = dlarray(alpha);
        learnable_parameters.linear.beta = dlarray(beta);

    elseif type_reduction_method == "NTv4"
        alpha = (rand(number_mf,number_outputs)*1)-0.5;
        beta = (rand(number_mf,number_outputs)*1)-0.5;
        learnable_parameters.linear.alpha = dlarray(alpha);
        learnable_parameters.linear.beta = dlarray(beta);
    end


elseif output_type == "IV"

    c = rand(number_mf,number_outputs)*0.01;
    learnable_parameters.IV.c = dlarray(c);

    delta = rand(number_mf,number_outputs)*0.01;
    learnable_parameters.IV.delta = dlarray(delta);

    % delta_1 = rand(number_mf,number_outputs)*0.01;
    % learnable_parameters.IV.delta_1 = dlarray(delta_1);
    
    % delta_2 = rand(number_mf,number_outputs)*0.01;
    % learnable_parameters.IV.delta_2 = dlarray(delta_2);
    
    if type_reduction_method == "BMM" || type_reduction_method == "NT_alpha" || type_reduction_method == "NT_alpha2" || type_reduction_method == "KM_BMM" || type_reduction_method == "CQTR"

        alpha = (rand(1,number_outputs)*2)-1;
        learnable_parameters.IV.alpha = dlarray(alpha);

    elseif type_reduction_method == "NT_multi_alpha"  || type_reduction_method == "CQTR_multi_alpha"
        alpha = (rand(number_mf,number_outputs)*2)-1;
        learnable_parameters.IV.alpha = dlarray(alpha);

    elseif type_reduction_method == "NTv2" || type_reduction_method == "NTv2_alpha2" || type_reduction_method == "NTv3" || type_reduction_method == "NTv3_alpha2"
        alpha = (rand(1,number_outputs)*2)-1;
        beta = (rand(1,number_outputs)*2)-1;
        learnable_parameters.IV.alpha = dlarray(alpha);
        learnable_parameters.IV.beta = dlarray(beta);

    elseif type_reduction_method == "NTv2_multi_alpha" || type_reduction_method == "NTv3_multi_alpha"
        alpha = (rand(number_mf,number_outputs)*2)-1;
        beta = (rand(number_mf,number_outputs)*2)-1;
        learnable_parameters.IV.alpha = dlarray(alpha);
        learnable_parameters.IV.beta = dlarray(beta);

    elseif type_reduction_method == "NTv4"
        alpha = (rand(number_mf,number_outputs)*1)-0.5;
        beta = (rand(number_mf,number_outputs)*1)-0.5;
        learnable_parameters.IV.alpha = dlarray(alpha);
        learnable_parameters.IV.beta = dlarray(beta);
    end




elseif output_type == "IVL"

    a = rand(number_mf*number_outputs,number_inputs)*0.01;
    b = rand(number_mf*number_outputs,1)*0.01; % single output


    delta_a = rand(number_mf*number_outputs,number_inputs)*0.01;
    delta_b = rand(number_mf*number_outputs,1)*0.01; % single output

    learnable_parameters.IVL.a = dlarray(a);
    learnable_parameters.IVL.delta_a = dlarray(delta_a);
    learnable_parameters.IVL.b = dlarray(b);
    learnable_parameters.IVL.delta_b = dlarray(delta_b);

    if type_reduction_method == "BMM" || type_reduction_method == "NT_alpha" || type_reduction_method == "NT_alpha2" || type_reduction_method == "KM_BMM" || type_reduction_method == "CQTR"
        alpha = (rand(1,number_outputs)*2)-1;
        learnable_parameters.IVL.alpha = dlarray(alpha);

    elseif type_reduction_method == "NT_multi_alpha"  || type_reduction_method == "CQTR_multi_alpha"
        alpha = (rand(number_mf,number_outputs)*2)-1;
        learnable_parameters.IVL.alpha = dlarray(alpha);

    elseif type_reduction_method == "NTv2" || type_reduction_method == "NTv2_alpha2" || type_reduction_method == "NTv3" || type_reduction_method == "NTv3_alpha2"
        alpha = (rand(1,number_outputs)*2)-1;
        beta = (rand(1,number_outputs)*2)-1;
        learnable_parameters.IVL.alpha = dlarray(alpha);
        learnable_parameters.IVL.beta = dlarray(beta);

    elseif type_reduction_method == "NTv2_multi_alpha" || type_reduction_method == "NTv3_multi_alpha"
        alpha = (rand(number_mf,number_outputs)*2)-1;
        beta = (rand(number_mf,number_outputs)*2)-1;
        learnable_parameters.IVL.alpha = dlarray(alpha);
        learnable_parameters.IVL.beta = dlarray(beta);

    elseif type_reduction_method == "NTv4"
        alpha = (rand(number_mf,number_outputs)*1)-0.5;
        beta = (rand(number_mf,number_outputs)*1)-0.5;
        learnable_parameters.IVL.alpha = dlarray(alpha);
        learnable_parameters.IVL.beta = dlarray(beta);
    end


end

end