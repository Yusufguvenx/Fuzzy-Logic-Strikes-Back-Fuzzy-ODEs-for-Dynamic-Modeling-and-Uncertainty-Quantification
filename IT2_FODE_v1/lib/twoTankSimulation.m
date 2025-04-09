function [X, U] = twoTankSimulation(seed)
    % Parameters
    c1 = 0.08; % inlet valve coefficient
    c2 = 0.04; % tank outlet coefficient

    % Initial conditions
    x0 = [0, 0]; % Initial levels of the tanks
    Ts = 1;
    nsim = 3000; % Number of simulation steps
    tspan = 0:Ts:(nsim - 1)*Ts;


    % Generate input signal
    U = step_(nsim, 2, 0, 0.4, ceil(nsim/100), seed);

    h1(1) = x0(1);
    h2(1) = x0(2);

    % Solve ODE
    % [t, x] = ode45(@(t,x) twoTankODE(t, x, u, c1, c2, tspan), tspan, x0);

    % Simulate CSTR
    for i = 1:length(tspan)-1
        ts = [tspan(i), tspan(i+1)];
        y = ode45(@(t, x) twoTankODE_2(t, x, U(i, :), c1, c2), ts, x0);
        h1(i+1) = y.y(1, end);
        h2(i+1) = y.y(2, end);
        x0(1) = h1(i+1);
        x0(2) = h2(i+1);
    end
    
    X = [h1;h2];
    U = U';
    

end

function dxdt = twoTankODE(t, x, u, c1, c2, tspan)
    % Interpolate the control input based on the current time
    timeSteps = tspan;
    pump = interp1(timeSteps, u(:,1), t, 'previous', 'extrap');
    valve = interp1(timeSteps, u(:,2), t, 'previous', 'extrap');

    % Tank levels cannot go below 0 or above 1
    h1 = max(min(x(1), 1), 0);
    h2 = max(min(x(2), 1), 0);
    
    % Differential equations
    dhdt1 = c1 * (1 - valve) * pump - c2 * sqrt(h1);
    dhdt2 = c1 * valve * pump + c2 * sqrt(h1) - c2 * sqrt(h2);
    
    % Prevent overflow
    if h1 >= 1 && dhdt1 > 0
        dhdt1 = 0;
    end
    if h2 >= 1 && dhdt2 > 0
        dhdt2 = 0;
    end
    
    dxdt = [dhdt1; dhdt2];
end

function dxdt = twoTankODE_2(t, x, u, c1, c2)
    % Interpolate the control input based on the current time

    pump = u(:,1);
    valve = u(:,2);

    % Tank levels cannot go below 0 or above 1
    h1 = max(min(x(1), 1), 0);
    h2 = max(min(x(2), 1), 0);
    
    % Differential equations
    dhdt1 = c1 * (1 - valve) * pump - c2 * sqrt(h1);
    dhdt2 = c1 * valve * pump + c2 * sqrt(h1) - c2 * sqrt(h2);
    
    % Prevent overflow
    if h1 >= 1 && dhdt1 > 0
        dhdt1 = 0;
    end
    if h2 >= 1 && dhdt2 > 0
        dhdt2 = 0;
    end
    
    dxdt = [dhdt1; dhdt2];
end


function signal = step_(nsim, d, min, max, randsteps, seed, values)
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
