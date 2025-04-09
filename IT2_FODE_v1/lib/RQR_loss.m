function rqr_loss = RQR_loss(y, y_lower,y_upper, cov, mbs)


% errors1 = permute(y - y_lower, [1 3 2]);
% errors2 = permute(y - y_upper, [1 3 2]);

errors1 = y - y_lower;
errors2 = y - y_upper;


loss = max(errors1.*errors2*(cov), errors1.*errors2*(cov-1));

% losses = [];
% 
% for k = 1:length(y)
% 
%     loss_ = max(errors1(:, :, k).*errors2(:, :, k)*(cov), errors1(:, :, k).*errors2(:, :, k)*(cov-1))
%     loss_ = mean(loss_, "all");
%     losses = [loss_ losses];
% 
% 
% end



rqr_loss = 1/mbs*sum(loss, "all");


end

