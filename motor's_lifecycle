% Load the dataset
data = load('motor_data.mat');

% Split the data into training and testing sets
train_ratio = 0.8;
num_samples = size(data, 1);
num_train_samples = round(train_ratio * num_samples);
train_indices = randsample(num_samples, num_train_samples);
test_indices = setdiff(1:num_samples, train_indices);

train_data = data(train_indices, :);
test_data = data(test_indices, :);

% Train a linear regression model
X = train_data(:, 1:2);
Y = train_data(:, 3);
model = fitlm(X, Y);

% Evaluate the model on the test set
X_test = test_data(:, 1:2);
Y_test = test_data(:, 3);
Y_pred = predict(model, X_test);

% Compute the root mean squared error
rmse = sqrt(mean((Y_pred - Y_test).^2));

% Make a prediction for a new motor
new_motor_age = 5;
new_motor_usage_hours = 10000;
new_motor_longevity = predict(model, [new_motor_age, new_motor_usage_hours]);

disp(['The predicted longevity for the new motor is ', num2str(new_motor_longevity)]);
