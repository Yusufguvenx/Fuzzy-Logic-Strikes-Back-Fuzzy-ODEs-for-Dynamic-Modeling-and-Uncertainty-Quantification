function y = clipped_relu(x, clip_threshold_lower , clip_threshold_upper)
y = min(max(x,clip_threshold_lower),clip_threshold_upper);
end