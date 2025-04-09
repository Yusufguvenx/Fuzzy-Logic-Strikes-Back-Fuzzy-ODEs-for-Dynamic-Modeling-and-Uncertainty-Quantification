function loss = lube_loss(y, ypred, y_lower, y_upper, timesteps, alpha)


% y = y(1, :, :);
% y_lower = y_lower(1, :, :);
% y_upper = y_upper(1, :, :);


y_max = max(y, [], 2);
y_min = min(y, [], 2);

range = y_max - y_min;


% mpiw_pen = (1/timesteps)*sum(abs(y_upper - y) + abs(y-y_lower), 2);

mpiw = sum(y_upper - y_lower, 2)/timesteps;

cond_l = y_lower > y;
n_l = sum(cond_l, 2);
cond_u = y_upper > y;
n_u = sum(cond_u, 2);

cov = (n_u - n_l) / timesteps;

loss = mpiw./range.*(1+exp(10*relu(1-alpha-cov)));

loss = mean(loss, "all");

end