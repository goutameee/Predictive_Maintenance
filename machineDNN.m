% Load the data
load('machine_data.mat');

% Define the DNN architecture
layers = [featureInputLayer(8)
          fullyConnectedLayer(16)
          reluLayer()
          fullyConnectedLayer(8)
          reluLayer()
          fullyConnectedLayer(1)
          regressionLayer()];

% Set the training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {Xval, Yval}, ...
    'Plots', 'training-progress');

% Train the DNN
net = trainNetwork(Xtrain, Ytrain, layers, options);

% Validate the DNN
Ypred_val = predict(net, Xval);
mse_val = mean((Yval - Ypred_val).^2);

% Test the DNN
Ypred_test = predict(net, Xtest);
mse_test = mean((Ytest - Ypred_test).^2);
