function tilted_loss = tilted_loss(y, y_lower, y_upper, q1, q2, mbs, alpha)


lower_loss = 1/mbs*(sum(max(q1*(y-y_lower), (q1-1)*(y-y_lower)), "all"));
upper_loss = 1/mbs*(sum(max(q2*(y-y_upper), (q2-1)*(y-y_upper)), "all"));

% %added
% L_losses = [];
% U_losses = [];
% 
% for k = 1:mbs
% l_loss = sum(max(q1*(y(:, :, k) - y_lower(:, :, k)), (q1-1)*(y(:, :, k) - y_lower(:, :, k))), 2);
% u_loss = sum(max(q2*(y(:, :, k) - y_upper(:, :, k)), (q2-1)*(y(:, :, k) - y_upper(:, :, k))), 2);
% L_losses = [L_losses, l_loss];
% U_losses = [U_losses, u_loss];
% 
% end



tilted_loss = lower_loss + upper_loss;

end

