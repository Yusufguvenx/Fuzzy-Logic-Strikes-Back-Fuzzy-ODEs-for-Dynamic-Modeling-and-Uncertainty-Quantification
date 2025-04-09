function rqrw_loss = RQRW_loss(y, y_lower,y_upper, cov, lambda, mbs)



errors1 = permute(y - y_lower, [1 3 2]);
errors2 = permute(y - y_upper, [1 3 2]);
errors = permute(y_upper - y_lower, [1 3 2]);

loss1 = max(errors1.*errors2*(cov + 2*lambda), errors1.*errors2*(cov + 2*lambda-1));
loss2 = 0.5*lambda*errors.^2;

rqrw_loss = 1/mbs*sum(loss1 + loss2, "all");

end

