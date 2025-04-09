
% In this example, we implement Neural ODE via nlssest function, MATLAB DL
% toolbox function. We used SI Engine Dataset for this experiment. After
% training, for simulation we did denormalization, and RMSE values are
% calculated with these denormalized outputs.

%% For reproducability

seed = 0;

%% SI Engine Dataset

dataset_name = 'SIEngine';

load datasets/SIEngineData IOData

eData = IOData(1:6e4,:);     % portion used for estimation
vData = IOData(6e4+1:end,:); % portion used for validation

[eDataN, C, S] = normalize(eData);
s_x = table2array(S);
c_x = table2array(C);

vDataN = (vData - c_x) ./ s_x;
Ts = 1;

% Downsample datasets 10 times
eDataD = idresamp(eDataN,[1 10]); 
vDataD = idresamp(vDataN,[1 10]);
%

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

Inputs = ["ThrottlePosition","WastegateValve","EngineSpeed","SparkTiming"];
Output = "EngineTorque";

xTest = table2array(vDataD);
out_val = xTest(:, end);
xTest(:, end) = [];
xTest = [out_val xTest];
xTest0 = xTest(1, 1);

uTest = xTest(:, 2:end);
uTest = permute(uTest, [2 1]);
tTest = 0.1:Ts:size(xTest, 1)*Ts;

xTest = xTest(:, 1)';%added

std1 = s_x(end);
mu1 = c_x(end);

number_of_epoch = 90;
mbs = 100;
lr = 1e-3;


%%

% Run your Neural ODE operation here

rng(seed)

nss = idNeuralStateSpace(1,NumInputs=4, Ts=Ts);
nss.InputName = Inputs;%added
nss.OutputName = Output;%added


nss.StateNetwork = createMLPNetwork(nss,'state', ...
    LayerSizes= [128 128], ...
    WeightsInitializer="glorot", ...
    BiasInitializer="zeros", ...
    Activations='tanh');

opt = nssTrainingOptions('adam');
opt.MaxEpochs = number_of_epoch;
opt.MiniBatchSize = mbs;

opt.LearnRate = lr;

nss = nlssest(Expts,nss,opt);%added

%% inference for Neural ODE

U = array2timetable(uTest',RowTimes=seconds(tTest'), VariableNames=Inputs);%added

% Simulate neural state-space system from x0
simOpt = simOptions('InitialCondition',xTest0);
yn = sim(nss,U,simOpt);

yPred = table2array(yn)';
yPreds_mean = yPred.*std1 + mu1;
yTestVal = xTest.*std1 + mu1;

%for visualization ground truth vs predicted estimations
plotPredictions(yTestVal, yPreds_mean)

err = yTestVal - yPreds_mean;

NRMSE = sqrt(sum(err.^2, 2)./sum((yTestVal-mean(yTestVal, 2)).^2, 2)); 
accuracy = 100*(1-NRMSE);

testRMSE = rmse(yTestVal, yPreds_mean, 2);








