%Perceptron Based learning with 2 inputs and 1 output
close all; clear; clc;
load(['Two_moons_overlap.mat'])%change this to desired plot

bias = 0;
inputX = [bias X(1,:)]; %[bias, x1, x2]
oldWeight = [1;0;0]; %[1,w1,w2]
newWeight = [1;0;0];
alpha = .9;
theta = 2;
error = zeros(1000,1);
y = zeros(1,1000);
for j = 1:1:1000 %checks if any errors at the beginning, should be but just to check
    if (y(j) ~= Y(j))
        error(j) = 1;
    end
end

while(sum(error) > 60 ) % there is # errors in error vector
    for i = 1:1:1000
        inputX = [bias X(i,:)];%update input
        oldWeight = newWeight; %set newest weight to old weight so it is used for calculation
        output = transpose(oldWeight) * transpose(inputX); %determines sum of weights and inputs
        if(output > theta) % determines activation ouput
            y(i) = 1;
        elseif(output < -theta)
            y(i) = -1;
        else
            y(i) = 0;
        end

        if(y(i) ~= Y(i))% update weights if error
            error(i) = 1;
            newWeight(2) = oldWeight(2) + alpha * Y(i) * inputX(2); %update w1
            newWeight(3) = oldWeight(3) + alpha * Y(i) * inputX(3); %update w2
            bias = bias + alpha * Y(i); %update bias

        else %no error in weights
            error(i) = 0;
            newWeight = oldWeight; % new = old
        end
        disp(sum(error))
    end
end

%boundary equations
x = -20:29;
y1 = (theta - bias - newWeight(2) * x) / newWeight(3); %greater than theta
y2 = (-theta - bias - newWeight(2) * x) / newWeight(3); %less than negative theta
figure(1)
scatter(X(:,1),X(:,2))
hold on
b1 = plot(x,y1); L1 = "Greater than theta";
b2 = plot(x,y2); L2 = "Less than negative theta";
legend('Moons',L1,L2);

