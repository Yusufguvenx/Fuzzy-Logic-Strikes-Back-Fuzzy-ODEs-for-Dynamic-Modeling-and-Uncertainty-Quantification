function [X, U] = cstr_data(seed, nsim, sec)



U = 300. + step(nsim, 1, -3, 3, ceil(nsim/25), seed);

% U = 300;

% Steady State Initial Conditions for the States
Ca_ss = 0.87725294608097;
T_ss = 324.475443431599;
x0 = [Ca_ss; T_ss];

% Time Interval (min)
% t = linspace(0, sec, nsim);
t = 0:0.1:sec;

% Store results for plotting
Ca = ones(size(t)) * Ca_ss;
T = ones(size(t)) * T_ss;
Tc = U';

% Step cooling temperature
% 
% PP1 = griddedInterpolant(t, permute(Tc, [2, 1, 3]), "previous");
% func = @(t, x) cstr_2(t, x, PP1);
% 
% [~, xhat] = ode45(func, t, x0);
% xhat = xhat';


% Simulate CSTR
for i = 1:length(t)-1
    ts = [t(i), t(i+1)];
    y = ode45(@(t, x) cstr(t, x, Tc(i)), ts, x0);
    Ca(i+1) = y.y(1, end);
    T(i+1) = y.y(2, end);
    x0(1) = Ca(i+1);
    x0(2) = T(i+1);
end

% X = xhat;
X = [Ca;T];
U = U';

end

%% Define CSTR model
function xdot = cstr(t, x, Tc)
    Ca = x(1);
    T = x(2);
    Tf = 350;
    Caf = 1.0;
    q = 100;
    V = 100;
    rho = 1000;
    Cp = 0.239;
    mdelH = 5e4;
    EoverR = 8750;
    k0 = 7.2e10;
    UA = 5e4;
    rA = k0*exp(-EoverR/T)*Ca;
    dCadt = q/V*(Caf - Ca) - rA;
    dTdt = q/V*(Tf - T) + mdelH/(rho*Cp)*rA + UA/V/rho/Cp*(Tc-T);
    xdot = [dCadt; dTdt];
end

%%
function xdot = cstr_2(t, x, PP)

    Tc = permute(PP(t), [2 3 1]);

    Ca = x(1);
    T = x(2);
    Tf = 350;
    Caf = 1.0;
    q = 100;
    V = 100;
    rho = 1000;
    Cp = 0.239;
    mdelH = 5e4;
    EoverR = 8750;
    k0 = 7.2e10;
    UA = 5e4;
    rA = k0*exp(-EoverR/T)*Ca;
    dCadt = q/V*(Caf - Ca) - rA;
    dTdt = q/V*(Tf - T) + mdelH/(rho*Cp)*rA + UA/V/rho/Cp*(Tc-T);
    xdot = [dCadt; dTdt];
end
%%
function signal = step(nsim, d, min, max, randsteps, seed, values)
    rng(seed)
    % Random step function for arbitrary number of dimensions

    % Arguments:
    % nsim: Number of simulation steps
    % d: Number of dimensions
    % min: Lower bound on values
    % max: Upper bound on values
    % randsteps: Number of random steps in time series (will infer from values if values is not None)
    % values: An ordered list of values for each step change

    % Ensure min and max are column vectors
    if numel(min) == 1
        min = min * ones(1, d);
    end
    if numel(max) == 1
        max = max * ones(1, d);
    end

    % Generate random steps if values are not provided
    if nargin < 7 || isempty(values)
        values = rand(randsteps, d) .* (max - min) + min;
    end

    % Repeat values to match nsim
    num_repeats = ceil(nsim / size(values, 1));
    signal = values(repelem(1:size(values, 1), num_repeats), :);


    % signal = repelem(values, num_repeats);
    signal = signal(1:nsim, :);
end

