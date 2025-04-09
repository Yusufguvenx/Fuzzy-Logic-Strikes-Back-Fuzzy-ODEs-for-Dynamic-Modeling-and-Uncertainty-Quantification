

% In this example, we implement IT2-Fuzzy ODE. During calculation of the
% loss function, we used the derivative-like quantity of IT2-Fuzzy ODE.
% We used SI Engine Dataset for this experiment. After
% training, for simulation we did denormalization, and RMSE
%  PCIP, and PINAW values are
% calculated with these denormalized outputs.

%% For reproducability

seed = 0;

%% SI Engine Data


dataset_name = 'SIEngine';

load datasets/SIEngineData IOData

eData = IOData(1:6e4,:);     % portion used for estimation
vData = IOData(6e4+1:end,:); % portion used for validation

[eDataN, C, S] = normalize(eData);
s_x = table2array(S);
c_x = table2array(C);

vDataN = (vData - c_x) ./ s_x;


% eData = IODataN(1:6e4,:);     % portion used for estimation
% vData = IODataN(6e4+1:end,:); % portion used for validation

% Downsample datasets 10 times
eDataD = idresamp(eDataN,[1 10]); 
vDataD = idresamp(vDataN,[1 10]);
Ts = 1;

predictionStep = 20; % length of each data segment
numExperiment = height(eDataD) - predictionStep;
Expts = cell(1, numExperiment); 

% this for is used for comparison with my data structure
for i = 1:numExperiment
    Expts{i} = eDataD(i:i+predictionStep,:);
    if i>1
       % set the row time of each segment to be identical; this is a requirement for training a
       % neural state-space model with multiple data experiments
       Expts{i}.Properties.RowTimes = Expts{1}.Properties.RowTimes;
    end
end


xTrain = table2array(eDataD);
out = xTrain(:, end);
xTrain(:, end) = [];
xTrain = [out xTrain];
xTrain0 = xTrain(1, 1);
yTrue = xTrain(:, 1)';


training_num = size(xTrain, 1);
t = 0.1:Ts:size(xTrain, 1)*Ts;
neuralOdeTimesteps = 20;
mbs = 100;
number_of_epoch = 90;
lr = 0.01;
learnRate = lr;

%Validation

xTest = table2array(vDataD);
out_val = xTest(:, end);
xTest(:, end) = [];
xTest = [out_val xTest];
xTest0 = xTest(1, 1);
yTest = xTest(:, 1)';

uTest = xTest(:, 2:end);
uTest = permute(uTest, [2 1]);
tTest = 0.1:Ts:size(xTest, 1)*Ts;

xTest = xTest(:, 1)';%added

std1 = s_x(end);
mu1 = c_x(end);
%%


number_mf = 10; % number of rules == number of membership functions
number_inputs = 5; % total column space, nx + nu
number_outputs = 1; % number of outputs, number of states, nx

input_membership_type = "gaussmf";
input_type ="HS";

output_membership_type = "linear";

type_reduction_method = "KM";




gradDecay = 0.9;
sqGradDecay = 0.999;

PI_values = [];
test_results = [];
PINAW_values = [];

plotFrequency = 50;



learnRate = lr;

averageGrad = [];
averageSqGrad = [];


close all

%%
rng(seed)
%%

if type_reduction_method == "KM" || type_reduction_method == "KM_BMM"
    u = int2bit(0:(2^number_mf)-1,number_mf);
else
    u = 0;
end

%% split by number ------------------------------
data = [xTrain];
data_size = size(data,1);
test_num = data_size-training_num;


Training_temp = data((1:training_num),:);

%% ------------------------------

%training data
Train.inputs = reshape(Training_temp(:,1:number_inputs)', [1, number_inputs, training_num]); % traspose come from the working mechanism of the reshape, so it is a must
Train.outputs = reshape(Training_temp(:,1:number_outputs)', [1, number_outputs, training_num]);

Train.inputs = dlarray(Train.inputs);


%% init

[Learnable_parameters, timesteps] = initialize_Glorot_IT2_dlode(Train.inputs, number_inputs, number_outputs, input_type,output_membership_type, number_mf, type_reduction_method, t, neuralOdeTimesteps);
prev_learnable_parameters = Learnable_parameters;

%% split data state and input

X = Train.inputs(:, 1:number_outputs, :);
ux = Train.inputs(:, number_outputs+1:end, :);
ux = permute(ux, [2 3 1]);
ux = extractdata(ux);

%% rng reset
rng(seed)

%%

number_of_iter_per_epoch = floorDiv(training_num-neuralOdeTimesteps, mbs);

number_of_iter = number_of_epoch * number_of_iter_per_epoch;
global_iteration = 1;

for epoch = 1: number_of_epoch

    [batch_inputs, U, batch_targets] = create_mini_batch(X, ux, neuralOdeTimesteps, training_num-neuralOdeTimesteps);

    % 
    for iter = 1:number_of_iter_per_epoch

    [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets, iter, mbs);
    [loss, gradients, yPred_train_lower, yPred_train_upper, yPred_train] = dlfeval(@IT2_ModelLoss_dlode, timesteps, mini_batch_inputs ,...
        number_inputs, u_mini_batch, targets,number_outputs, number_mf, mbs, Learnable_parameters, output_membership_type,...
        input_membership_type,input_type,type_reduction_method,u, neuralOdeTimesteps);

    [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad,...
        epoch, learnRate, gradDecay, sqGradDecay);

    end

    % %testing in each epoch
    if(epoch==1 || mod(epoch, plotFrequency) == 0)
        x = xTrain0;
        yPred = x';
        ahead = length(t)-1;
        PP = griddedInterpolant(t, permute(ux,[2, 1, 3]), "pchip");
        Ux = permute(PP(t(:)),[2 3 1]);
        X_lower = dlarray(zeros(size(x,2) ,ahead, size(x,1)));
        X_upper = dlarray(zeros(size(x,2) ,ahead, size(x,1)));
        X_mean = dlarray(zeros(size(x,2) ,ahead, size(x,1)));

        for ct = 1:ahead
            u_mini_batch = Ux(:, :, ct);
            [x_l, x_u, x_m] = evaluatemodel(x, u_mini_batch, Learnable_parameters,number_mf, number_inputs,number_outputs, mbs, output_membership_type, input_membership_type, input_type, type_reduction_method, u);
            X_lower(:, ct) = x_l';
            X_upper(:, ct) = x_u';
            X_mean(:, ct) = x_m';
            x = x_m;
        end

        yPreds_mean = [yPred X_mean];
        yPreds_lower = [yPred X_lower];
        yPreds_upper = [yPred X_upper];

    end


    plotter(epoch,plotFrequency,loss,yTrue,yPreds_mean,yPreds_upper,yPreds_lower);

end

%% Inference

x = xTest0;
yPred = x';
ahead = length(tTest)-1;

PP = griddedInterpolant(tTest, permute(uTest,[2, 1, 3]), "pchip");
Ux = permute(PP(tTest(:)),[2 3 1]);
X_lower = dlarray(zeros(size(x,2) ,ahead, size(x,1)));
X_upper = dlarray(zeros(size(x,2) ,ahead, size(x,1)));
X_mean = dlarray(zeros(size(x,2) ,ahead, size(x,1)));

for ct = 1:ahead
    u_mini_batch = Ux(:, :, ct);
    [x_l, x_u, x_m] = evaluatemodel(x, u_mini_batch, Learnable_parameters,number_mf, number_inputs,number_outputs, mbs, output_membership_type, input_membership_type, input_type, type_reduction_method, u);
    X_lower(:, ct) = x_l';
    X_upper(:, ct) = x_u';
    X_mean(:, ct) = x_m';
    x = x_m;
end

yPreds_mean = [yPred X_mean];
yPreds_lower = [yPred X_lower];
yPreds_upper = [yPred X_upper];

yPreds_lower = yPreds_lower.*std1 + mu1;
yPreds_upper = yPreds_upper.*std1 + mu1;
yPreds_mean = yPreds_mean.*std1 + mu1;
yTestVal = xTest.*std1 + mu1;

plotPredictions(yTestVal, yPreds_mean, yPreds_lower, yPreds_upper)

err = yTestVal - yPreds_mean;

NRMSE = sqrt(sum(err.^2, 2)./sum((yTestVal-mean(yTestVal, 2)).^2, 2)); 
accuracy = 100*(1-NRMSE);


testRMSE = rmse(yTestVal, yPreds_mean, 2);



PICP_test = PICP(yTestVal, yPreds_lower, yPreds_upper);
PINAW_test = PINAW(yTestVal, yPreds_lower, yPreds_upper);

%%
function [X0, U, targets]  = create_mini_batch(X, ux, ahead, numexamples)

X = permute(X, [2, 3, 1]);
% ux = permute(ux, [2 3 1]);

shuffle_idx = randperm(size(X, 2)-ahead);

X0 = dlarray(X(:, shuffle_idx));
targets = dlarray(zeros([size(X, 1) ahead, numexamples]));
U = zeros([size(ux, 1), ahead+1, numexamples]);

for i =1:numexamples
    targets(:, :, i) = X(:, shuffle_idx(i) + 1: shuffle_idx(i) + ahead);
     U(:, :, i) = ux(:, shuffle_idx(i): shuffle_idx(i) + ahead);
end

X0 = permute(X0, [3 1 2]);

end

%%
function [X0, U, targets]  = create_mini_batch_new(X, ux, ahead, numexamples)

X = permute(X, [2, 3, 1]);
% ux = permute(ux, [2 3 1]);

numExperiment = length(X) - ahead;
targets = dlarray(zeros([size(X, 1) ahead, numExperiment]));
X0 = dlarray(zeros([size(X, 1), numExperiment]));
U = zeros([size(ux, 1), ahead+1, numexamples]);

for i = 1:numExperiment
    X0(:, i) = X(:, i);
    targets(:, :, i) = X(:, i+1:i+ahead);
    U(:, :, i) = ux(:, i:i+ahead);
end

X0 = permute(X0, [3 1 2]);


end
%%
function [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets,iter,mbs)

mini_batch_inputs = batch_inputs(:, :, ((iter-1)*mbs)+1:(iter*mbs));
targets = batch_targets(:, :, ((iter-1)*mbs)+1:(iter*mbs));
u_mini_batch = U(:, :, ((iter-1)*mbs)+1:(iter*mbs));

end