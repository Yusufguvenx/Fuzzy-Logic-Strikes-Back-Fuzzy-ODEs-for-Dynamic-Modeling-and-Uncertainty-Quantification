function [X, U] = cstr_data_2(seed)

Ca_ss = 0.87725294608097;
T_ss = 324.475443431599;
x0 = [Ca_ss; T_ss];

Ts = 0.1;
nsim = 3001; % Number of simulation steps
tspan = 0:Ts:(nsim - 1)*Ts;

U = 300. + step(nsim, 1, -3, 3, ceil(nsim/25), seed);

[t, x] = ode45(@(t,x) cstr_2(t, x, U, tspan), tspan, x0);


% Step cooling temperature
% 
% PP1 = griddedInterpolant(t, permute(Tc, [2, 1, 3]), "previous");
% func = @(t, x) cstr_2(t, x, PP1);
% 
% [~, xhat] = ode45(func, t, x0);
% xhat = xhat';

X = x';
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
function xdot = cstr_2(t, x, u, tspan)

    timeSteps = tspan;
    Tc = interp1(timeSteps, u(:,1), t, 'previous', 'extrap');
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

