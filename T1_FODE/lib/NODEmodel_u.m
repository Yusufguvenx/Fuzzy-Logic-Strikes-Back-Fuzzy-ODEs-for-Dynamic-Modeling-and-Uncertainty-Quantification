function X = NODEmodel_u(t, number_mf, mini_batch_inputs, ux, number_inputs, number_outputs, learnable_parameters, mbs, output_membership_type)
x0 = mini_batch_inputs;
x = x0;

ahead = length(t)-1;
PP = griddedInterpolant(t, permute(ux,[2, 1, 3]), "spline");
Ux = permute(PP(t(:)),[2 3 1]);
X = dlarray(zeros(size(x,2) ,ahead, size(x,3)));

for ct = 1:ahead
    u = Ux(:,:,ct);
    x = fismodel_u(x0, u, number_mf, number_inputs,number_outputs, mbs, learnable_parameters, output_membership_type); 
    x = permute(x, [2, 1, 3]);
    X(:, ct, :) = x;
    x0 = permute(x, [2, 1, 3]);
end

end
