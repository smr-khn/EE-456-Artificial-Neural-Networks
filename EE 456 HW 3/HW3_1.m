% heteroassociative neural net using the Hebb rule
%Part A
load('SandT_patterns_HW3_P1.mat')
inputSize = size(S); % matrix size of input
outputSize = size(T); % matrix size of output

w = zeros(inputSize(1),inputSize(2),outputSize(2),outputSize(1)); %Matrix of weights for each input, each weight vector is a 7x9 mapped to the input indexes 9x7, 9x7x7x9
w_total = zeros(inputSize(1),inputSize(2),outputSize(2),outputSize(1)); % weights after adding up all input output wight matrices
%disp(size(w))
for k = 1:inputSize(3) % loop through all sets
    for i = 1:inputSize(1)
        for j = 1:inputSize(2)
            w(i,j,:,:) = S(i,j,k) * transpose(T(:,:,k)); %single weight matrix is S *T'
        end
    end
    w_total = w_total + w; % add all weight matrices being created
end

y = zeros(outputSize);% set output of net to size of T

for k = 1:inputSize(3) % loop through all inputs
    for i = 1:outputSize(1) % loop through all weight vectors
        for j = 1:outputSize(2)
            %disp(w_total(2,1,:,:))
            w_index = reshape(w_total(i,j,:,:),7,9);% reshape individual weight vector becasue matlab did not like it
            %disp(w_index)
            y(:,:,k) = transpose(w_index) * S(i,j,k); % y_in = W' * X
        end
    end

    for i = 1:outputSize(1) % loop through all values of y_in and change to bipolar
        for j = 1:outputSize(2)
            if (y(i,j,k) >= 0)
                y(i,j,k) = 1;
            else
                y(i,j,k) = -1;
            end
        end
    end
end
%check recall
%left is input, right is output
figure(1)
subplot(3,2,1),imagesc(T(:,:,1))
subplot(3,2,2),imagesc(y(:,:,1))
subplot(3,2,3),imagesc(T(:,:,2))
subplot(3,2,4),imagesc(y(:,:,2))
subplot(3,2,5),imagesc(T(:,:,3))
subplot(3,2,6),imagesc(y(:,:,3))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part B
%new input output pairs
newInputs = -1 * T(:,:,1:2);
newOutputs = -1 * S(:,:,2:3);
S = cat(3,S,newInputs);
T = cat(3,T,newOutputs);
outputSize = size(T); % matrix size of output
inputSize = size(S);

w = zeros(inputSize(1),inputSize(2),outputSize(2),outputSize(1)); %Matrix of weights for each input, each weight vector is a 7x9 mapped to the input indexes 9x7, 9x7x7x9
w_total = zeros(inputSize(1),inputSize(2),outputSize(2),outputSize(1)); % weights after adding up all input output wight matrices
%disp(size(w))
for k = 1:inputSize(3) % loop through all sets
    for i = 1:inputSize(1)
        for j = 1:inputSize(2)
            w(i,j,:,:) = S(i,j,k) * transpose(T(:,:,k)); %single weight matrix is S *T'
        end
    end
    w_total = w_total + w; % add all weight matrices being created
end

y = zeros(outputSize);% set output of net to size of T

for k = 1:inputSize(3) % loop through all inputs
    for i = 1:outputSize(1) % loop through all weight vectors
        for j = 1:outputSize(2)
            %disp(w_total(2,1,:,:))
            w_index = reshape(w_total(i,j,:,:),7,9);% reshape individual weight vector becasue matlab did not like it
            %disp(w_index)
            y(:,:,k) = transpose(w_index) * S(i,j,k); % y_in = W' * X
        end
    end

    for i = 1:outputSize(1) % loop through all values of y_in and change to bipolar
        for j = 1:outputSize(2)
            if (y(i,j,k) >= 0)
                y(i,j,k) = 1;
            else
                y(i,j,k) = -1;
            end
        end
    end
end
%check recall
figure(2)
subplot(5,2,1),imagesc(T(:,:,1))
subplot(5,2,2),imagesc(y(:,:,1))
subplot(5,2,3),imagesc(T(:,:,2))
subplot(5,2,4),imagesc(y(:,:,2))
subplot(5,2,5),imagesc(T(:,:,3))
subplot(5,2,6),imagesc(y(:,:,3))
subplot(5,2,7),imagesc(T(:,:,4))
subplot(5,2,8),imagesc(y(:,:,4))
subplot(5,2,9),imagesc(T(:,:,5))
subplot(5,2,10),imagesc(y(:,:,5))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part C
%get 12 random pixels
pixels = randsample(1:63, 12);
%include mssing and errors
% input 0's in first 6 indices of all images and errors in second 6 indices
% for all images
for i = 1:size(pixels,2)
    for j = 1:inputSize(3)
        if i >= 6
            S((pixels(i) + 63 * (j -1))) = 0;
        else
            S((pixels(i) + 63 * (j -1))) = -1 * S((pixels(i) + 63 * (j -1)));
        end
    end
end

for k = 1:inputSize(3) % loop through all inputs
    for i = 1:outputSize(1) % loop through all weight vectors
        for j = 1:outputSize(2)
            %disp(w_total(2,1,:,:))
            w_index = reshape(w_total(i,j,:,:),7,9);% reshape individual weight vector becasue matlab did not like it
            %disp(w_index)
            y(:,:,k) = transpose(w_index) * S(i,j,k); % y_in = W' * X
        end
    end

    for i = 1:outputSize(1) % loop through all values of y_in and change to bipolar
        for j = 1:outputSize(2)
            if (y(i,j,k) >= 0)
                y(i,j,k) = 1;
            else
                y(i,j,k) = -1;
            end
        end
    end
end
%check recall
figure(3)
subplot(5,2,1),imagesc(T(:,:,1))
subplot(5,2,2),imagesc(y(:,:,1))
subplot(5,2,3),imagesc(T(:,:,2))
subplot(5,2,4),imagesc(y(:,:,2))
subplot(5,2,5),imagesc(T(:,:,3))
subplot(5,2,6),imagesc(y(:,:,3))
subplot(5,2,7),imagesc(T(:,:,4))
subplot(5,2,8),imagesc(y(:,:,4))
subplot(5,2,9),imagesc(T(:,:,5))
subplot(5,2,10),imagesc(y(:,:,5))
