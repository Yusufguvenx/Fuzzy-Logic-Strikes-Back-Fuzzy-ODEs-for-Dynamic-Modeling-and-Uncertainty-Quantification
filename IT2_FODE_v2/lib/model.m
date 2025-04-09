function [dX_lower, dX_upper, dX] = model(t, mini_batch_inputs,  ux, number_mf, number_inputs,number_outputs, mbs, learnable_parameters, output_membership_type, input_mf_type, input_type,type_reduction_method,u)

x0 = mini_batch_inputs;
x = x0;

ahead = length(t)-1;
PP = griddedInterpolant(t, permute(ux,[2, 1, 3]), "pchip");
Ux = permute(PP(t(:)),[2 3 1]);
dX = dlarray(zeros(size(x,2) ,ahead, size(x,3)));
dX_lower = dlarray(zeros(size(x,2) ,ahead, size(x,3)));
dX_upper = dlarray(zeros(size(x,2) ,ahead, size(x,3)));

for ct = 1:ahead
    u_mini_batch = Ux(:,:,ct);
    [dx_lower, dx_upper, dx_mean] = odemodel(x0, u_mini_batch, learnable_parameters, number_mf, number_inputs, number_outputs, mbs, output_membership_type, input_mf_type, input_type, type_reduction_method, u); 
    
    dx_mean = permute(dx_mean, [2 1 3]);
    dx_lower = permute(dx_lower, [2 1 3]);
    dx_upper = permute(dx_upper, [2 1 3]);

    dX(:, ct, :) = dx_mean;
    dX_lower(:, ct, :) = dx_lower;
    dX_upper(:, ct, :) = dx_upper;

    x0 = permute(x0, [2 1 3]);
    x0 = x0 + dx_mean; %x(k+1) = x(k) + dx
    x0 = permute(x0, [2 1 3]);
end

end


