function y = custom_sign(x)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

x = x - 0.5;

if any(x == 0)
    x(x==0) = x(x==0) + 1e-6;
end

y = x./abs(x);
y = (y+1)./2;


end