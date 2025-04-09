function [x_norm,minimum,range] = max_min_norm(x,minimum,range)

arguments
    x = [];
    minimum = [];
    range = [];
end

eps = 1e-8;

if isempty(minimum)
minimum = min(x);
end

if isempty(range)
range = max(x) - minimum;
end

x_norm = (x-minimum)./(range+eps);
end