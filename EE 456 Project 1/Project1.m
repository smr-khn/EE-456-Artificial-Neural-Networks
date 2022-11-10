%Back Propagation example with 2 input, 20 hidden, 1 output nodes
%activation fun: tanh
%Threshold: 0
%Anneal: 10^-1 -> 10^-5

clc; clearvars, close all
load('DataSet1_MP1.mat')
load('DataSet2_MP1.mat')
addpath("functions")

%Data Set 1
data = [DataSet1 DataSet1_targets];
%Data Set 2
%data = [DataSet2 DataSet2_targets];
%shuffle data
shuffled_data = data(randperm(size(data, 1)), :);
%splitting data into training and testing subsets
training_input = shuffled_data(1:(.8 * size(shuffled_data,1)),1:2); %estimation subset 80% of training
test_input = shuffled_data((.8 * size(shuffled_data,1))+1:end,1:2); % validation subset
train_target = shuffled_data(1:(.8 * size(shuffled_data,1)),3); % target values for estimation
test_target = shuffled_data((.8 * size(shuffled_data,1))+1:end,3); %target values for validation

%initialize other variables
eta = .1; %anneal rate
threshold = 0; %threshold
w_hidden = rand(20,2); %2 inputs, 20 hidden = 40 weights
bias_hidden = ones(20,1) * .1; % 20 bias inputs for 20 hidden nodes
w_output = rand(20,1); % 20 hidden, 1 output = 20 weights
bias_output = .1; % bias for output node
output_err = 0; % back prop error from output
hidden_err = 0; % back prop error at hidden nodes

nn_size = [2,20,1]; % size of entire neural network for easy reference,
epoch = 1; % number of epochs
maxEpoch = 2000; % stopping criteria for epochs
training_error = zeros(1,maxEpoch); %variables to hold errors
test_error = zeros(1,maxEpoch);
while(epoch < maxEpoch) % stopping condition
    disp(epoch)
    for num = 1:size(training_input,1)
        %function to feed forward 1 pair of inputs and return the outputs
        [y, y_in, z, z_in]= feedForward(training_input(num,:),w_hidden,w_output,bias_hidden,bias_output,nn_size);
        %function to back propagate the output to hidden weights
        [output_err, delta_w_output, delta_bias_output] = backPropagationOutput(train_target(num),y,y_in,z,eta);
        %function to back propagate the hidden to input weights
        [delta_w_hidden, delta_bias_hidden] = backPropagationHidden(output_err,w_output,training_input(num,:),z_in,eta);
        %update weights with deltas calculated
        w_output = w_output + delta_w_output;
        bias_output = bias_output + delta_bias_output;
        w_hidden = w_hidden + delta_w_hidden;
        bias_hidden = bias_hidden + delta_bias_hidden;
        %calculate error
        training_error_energy = calculateError(y,train_target(num));
        training_error(epoch) = training_error(epoch) + training_error_energy;
    end
    training_error(epoch) = training_error(epoch) / size(training_input,1); % average error

    if(mod(epoch,4) == 0) % every 4 epochs, test against validation data
        incorrect = 0;
        for j = 1:size(test_input,1)
            %feed forward the validation/test data
            y_test= feedForward(test_input(j,:),w_hidden,w_output,bias_hidden,bias_output,nn_size);
            %calculate error
            test_error_energy = calculateError(y_test,test_target(j));
            test_error(epoch) = test_error(epoch) + test_error_energy;
            if(y_test * test_target(j) < 0)
                incorrect = incorrect + 1;
            end
        end
        test_error(epoch) = test_error(epoch) / size(test_input,1); % average error
    end
    epoch = epoch +1;
    annealCase = mod(epoch,500); % lower anneal rate every 500 epochs
    if (annealCase == 0)
        eta = eta * .1; %reduce by power of 10
    end
end

figure(1)
plot(1:maxEpoch,training_error)
hold on
plot(4:4:maxEpoch,test_error(4:4:end))
title('Training error and Validation error vs. Number of epochs')
xlabel('Epochs')
ylabel('Average Error')
legend('Training Error','Test Error')

figure(2)
scatter(test_input(:,1),test_input(:,2))
[X,Y] = meshgrid(-15:.1:25,-10:.1:15);
Z1 = zeros(size(X,1), size(X,2));
for i = 1:size(X,1)
    for j = 1:size(X,2)
        Z1(i,j)= feedForward([X(i,j) Y(i,j)],w_hidden,w_output,bias_hidden,bias_output,nn_size);
    end
end
Z = tanh(Z1);
hold on
contour(X,Y,Z)
title('Validation Set and Boundary Line')

%display error rate
sprintf('Overall Error Rate: %d', incorrect/size(test_input,1))
