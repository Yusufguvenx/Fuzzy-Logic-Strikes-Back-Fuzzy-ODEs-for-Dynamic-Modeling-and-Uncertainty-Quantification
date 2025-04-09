function Loss_QD = pearce_loss(y_upper, y_lower, y, lambda, softening_factor, alpha)
n = length(y);
gamma_U = sigmoid((y_upper-y) * softening_factor);
gamma_L = sigmoid((y-y_lower) * softening_factor);
gamma_ = gamma_U.*gamma_L;

gamma_U_hard = max(0, sign(y_upper-y));
gamma_L_hard = max(0, sign(y-y_lower));
gamma_hard = gamma_U_hard.*gamma_L_hard;


PICP_soft = mean(gamma_);

qd_rhs_soft = lambda*sqrt(n)*max(0, 1-alpha-PICP_soft)^2;
qd_lhs_hard = sum(abs(y_upper-y_lower).*gamma_hard) / (sum(gamma_hard) + 0.001);

Loss_QD = qd_lhs_hard + qd_rhs_soft;

end