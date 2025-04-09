function X = NODEmodel(t, number_mf, mini_batch_inputs, number_inputs, number_outputs, learnable_parameters, mbs, output_membership_type)
x = mini_batch_inputs;
ahead = length(t)-1;
% Ux = permute(interp1(t, permute(ux,[2 1 3]), t, "previous"), [2 3 1]);
X = dlarray(zeros(size(x,2) ,ahead, size(x,3)));

for ct = 1:ahead
    x = fismodel(x, number_mf, number_inputs,number_outputs, mbs, learnable_parameters, output_membership_type); 
    x = permute(x, [2, 1, 3]);
    X(:, ct, :) = x;
    x = permute(x, [2, 1, 3]);
end

end
