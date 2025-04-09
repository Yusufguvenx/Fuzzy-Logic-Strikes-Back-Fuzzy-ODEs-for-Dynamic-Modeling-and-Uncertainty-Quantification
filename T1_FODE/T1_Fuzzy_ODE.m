
% In this example, we implement T1-Fuzzy ODE.
% We used SI Engine Dataset for this experiment. After
% training, for simulation we did denormalization, and RMSE values are
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
current_path = pwd;
number_mf = 10;

number_inputs = 5; % total column space, nx + nu
number_outputs = 1; % number of outputs, nx, number of states

input_membership_type = "gaussmf";

output_membership_type = "linear";


gradDecay = 0.9;
sqGradDecay = 0.999;

test_results = [];

plotFrequency = 50;


learnRate = lr;

averageGrad = [];
averageSqGrad = [];

close all
%%
rng(seed)


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

[Learnable_parameters, timesteps] = initialize_Glorot_Kmeans(Train.inputs, Train.outputs, number_mf, output_membership_type, t, neuralOdeTimesteps);
prev_learnable_parameters = Learnable_parameters;

%% data manupulation

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


for iter = 1:number_of_iter_per_epoch

    [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets, iter, mbs);

    [loss, gradients, yPred_train] = dlfeval(@NODEfisModelLoss_u, timesteps, mini_batch_inputs, number_inputs, u_mini_batch, targets,number_outputs, number_mf, mbs, Learnable_parameters, output_membership_type);


    [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad,...
        epoch, learnRate, gradDecay, sqGradDecay);

end
% 
if(epoch==1 || mod(epoch, plotFrequency) == 0)

    x0 = xTrain0;
    yPred = x0';
    ahead = length(t)-1;
    PP = griddedInterpolant(t, permute(ux,[2, 1, 3]), "pchip");
    Ux = permute(PP(t(:)),[2 3 1]);
    X_mean = dlarray(zeros(size(x0,2) ,ahead, size(x0,1)));

    for ct = 1:ahead
        u = Ux(:,:,ct);
        x = fismodel_u(x0, u, number_mf, number_inputs,number_outputs, mbs, Learnable_parameters, output_membership_type); 
        X_mean(:, ct, :) = x';
        x0 = x;
    end

    yPreds_mean = [yPred X_mean];

end

plotter(epoch,plotFrequency,loss,yTrue, yPreds_mean);


end


%% Inference-discrete

x0 = xTest0;
yPred = x0';
ahead = length(tTest)-1;
PP = griddedInterpolant(tTest, permute(uTest,[2, 1, 3]), "pchip");
Ux = permute(PP(tTest(:)),[2 3 1]);
X_mean = dlarray(zeros(size(x0,2) ,ahead, size(x0,1)));

for ct = 1:ahead
    u = Ux(:,:,ct);
    x = fismodel_u(x0, u, number_mf, number_inputs,number_outputs, mbs, Learnable_parameters, output_membership_type); 
    X_mean(:, ct, :) = x';
    x0 = x;
end

yPreds_mean = [yPred X_mean];

yPreds_mean = yPreds_mean.*std1 + mu1;
yTestVal = xTest.*std1 + mu1;


plotPredictions(yTestVal, yPreds_mean)

err = yTestVal - yPreds_mean;

NRMSE = sqrt(sum(err.^2, 2)./sum((yTestVal-mean(yTestVal, 2)).^2, 2)); 
accuracy = 100*(1-NRMSE);


testRMSE = rmse(yTestVal, yPreds_mean, 2);




%%
function [X0, U, targets]  = create_mini_batch(X, ux,  ahead, numexamples)

X = permute(X, [2, 3, 1]);

shuffle_idx = randperm(size(X, 2)-ahead);

X0 = dlarray(X(:, shuffle_idx));
targets = dlarray(zeros([size(X, 1) ahead, numexamples]));
U = (zeros([size(ux, 1), ahead+1, numexamples]));

for i =1:numexamples
    targets(:, :, i) = X(:, shuffle_idx(i) + 1: shuffle_idx(i) + ahead);
     U(:, :, i) = ux(:, shuffle_idx(i): shuffle_idx(i) + ahead);
end

X0 = permute(X0, [3 1 2]);

end


%%
function [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets,iter,mbs)

mini_batch_inputs = batch_inputs(:, :, ((iter-1)*mbs)+1:(iter*mbs));
targets = batch_targets(:, :, ((iter-1)*mbs)+1:(iter*mbs));
u_mini_batch = U(:, :, ((iter-1)*mbs)+1:(iter*mbs));

end



