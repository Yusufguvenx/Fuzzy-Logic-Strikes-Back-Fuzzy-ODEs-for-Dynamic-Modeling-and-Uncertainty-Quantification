function tilted_loss = tilted_loss(y, y_lower, y_upper, q1, q2, mbs, alpha)


lower_loss = 1/mbs*(sum(max(q1*(y-y_lower), (q1-1)*(y-y_lower)), "all"));
upper_loss = 1/mbs*(sum(max(q2*(y-y_upper), (q2-1)*(y-y_upper)), "all"));



tilted_loss = lower_loss + upper_loss;

end

