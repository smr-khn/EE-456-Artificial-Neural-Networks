%autoassociative NN using the Hebb rule
%Part A
% heteroassociative neural net using the Hebb rule
%Part A
load('S_patterns_HW3_P2.mat')
input = zeros(63,1,10);
%turn matrices into 63x1 vectors
input(:,:,1) = reshape(S(:,:,1),63,1);
input(:,:,2) = reshape(S(:,:,2),63,1);

input(:,:,3) = reshape(S(:,:,3),63,1);
input(:,:,4) = reshape(S(:,:,4),63,1);

input(:,:,5) = reshape(S(:,:,5),63,1);
input(:,:,6) = reshape(S(:,:,6),63,1);
input(:,:,7) = reshape(S(:,:,7),63,1);
input(:,:,8) = reshape(S(:,:,8),63,1);
input(:,:,9) = reshape(S(:,:,9),63,1);
input(:,:,10) = reshape(S(:,:,10),63,1);

inputSize = size(input); % matrix size of input

w = zeros(63,63); %Matrix of weights for each input, each weight vector is a 7x9 mapped to the input indexes 9x7, 9x7x7x9
w_total = zeros(63,63); % weights after adding up all input output wight matrices
%disp(size(w))
for k = 1:inputSize(3) % loop through all sets
    w = input(:,:,k) * input(:,:,k)'; %single weight matrix is S *T'
    w_total = w_total + w; % add all weight matrices being created
end

y = zeros(63,1,inputSize(3));% set output of net to size of output

for k = 1:inputSize(3) % loop through all inputs
    y(:,:,k) = transpose(w_total)' * input(:,:,k); % y_in = W' * X
    for i = 1:inputSize(1) % loop through all values of y_in and change to bipolar
        for j = 1:inputSize(2)
            if (y(i,j,k) >= 0)
                y(i,j,k) = 1;
            else
                y(i,j,k) = -1;
            end
        end
    end
end
%reformat to 9x7
Y = zeros(9,7,inputSize(3));
Y(:,:,1) = reshape(y(:,:,1),9,7);
Y(:,:,2) = reshape(y(:,:,2),9,7);

Y(:,:,3) = reshape(y(:,:,3),9,7);
Y(:,:,4) = reshape(y(:,:,4),9,7);

Y(:,:,5) = reshape(y(:,:,5),9,7);
Y(:,:,6) = reshape(y(:,:,6),9,7);
Y(:,:,7) = reshape(y(:,:,7),9,7);
Y(:,:,8) = reshape(y(:,:,8),9,7);
Y(:,:,9) = reshape(y(:,:,9),9,7);
Y(:,:,10) = reshape(y(:,:,10),9,7);

%check recall

figure(1)
subplot(5,2,1),imagesc(S(:,:,1))
subplot(5,2,2),imagesc(Y(:,:,1))
subplot(5,2,3),imagesc(S(:,:,2))
subplot(5,2,4),imagesc(Y(:,:,2))
subplot(5,2,5),imagesc(S(:,:,3))
subplot(5,2,6),imagesc(Y(:,:,3))
subplot(5,2,7),imagesc(S(:,:,4))
subplot(5,2,8),imagesc(Y(:,:,4))
subplot(5,2,9),imagesc(S(:,:,5))
subplot(5,2,10),imagesc(Y(:,:,5))

figure(2)
subplot(5,2,1),imagesc(S(:,:,6))
subplot(5,2,2),imagesc(Y(:,:,6))
subplot(5,2,3),imagesc(S(:,:,7))
subplot(5,2,4),imagesc(Y(:,:,7))
subplot(5,2,5),imagesc(S(:,:,8))
subplot(5,2,6),imagesc(Y(:,:,8))
subplot(5,2,7),imagesc(S(:,:,9))
subplot(5,2,8),imagesc(Y(:,:,9))
subplot(5,2,9),imagesc(S(:,:,10))
subplot(5,2,10),imagesc(Y(:,:,10))


%Part A: The NN seems to not be able to recall any outputs. If you lower
%the set size to1,2, or 3, there is perfect recall

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Part B
% turn images into 63x1 vectors
input = zeros(63,1,10);
input(:,:,1) = reshape(S(:,:,1),63,1);
input(:,:,2) = reshape(S(:,:,2),63,1);
input(:,:,3) = reshape(S(:,:,3),63,1);
input(:,:,4) = reshape(S(:,:,4),63,1);
input(:,:,5) = reshape(S(:,:,5),63,1);
input(:,:,6) = reshape(S(:,:,6),63,1);
input(:,:,7) = reshape(S(:,:,7),63,1);
input(:,:,8) = reshape(S(:,:,8),63,1);
input(:,:,9) = reshape(S(:,:,9),63,1);
input(:,:,10) = reshape(S(:,:,10),63,1);
inputSize = size(input); % matrix size of input

w = zeros(63,63); %Matrix of weights for each input, each weight vector is a 7x9 mapped to the input indexes 9x7, 9x7x7x9
w_total = zeros(63,63); % weights after adding up all input output wight matrices
%disp(size(w))
for k = 1:inputSize(3) % loop through all sets
    w = input(:,:,k) * input(:,:,k)'; %single weight matrix is S *T'
    w_total = w_total + w; % add all weight matrices being created
end

y = zeros(63,1,10);% set output of net to size of output

%adding in noise
%S(1,:,:) = -1 * S(1,:,:);
%S(:,2,:) = -1 * S(:,2,:);
%S(2,:,:) = -1 * S(2,:,:);
%S(:,4,:) = -1 * S(:,4,:);
%S(:,5,:) = -1 * S(:,5,:);
%S(:,6,:) = -1 * S(:,6,:);

for k = 1:inputSize(3) % loop through all inputs
    y(:,:,k) = transpose(w_total)' * input(:,:,k); % y_in = W' * X
    for i = 1:inputSize(1) % loop through all values of y_in and change to bipolar
        for j = 1:inputSize(2)
            if (y(i,j,k) >= 0)
                y(i,j,k) = 1;
            else
                y(i,j,k) = -1;
            end
        end
    end
end
%reformat
Y = zeros(9,7,10);
Y(:,:,1) = reshape(y(:,:,1),9,7);
Y(:,:,2) = reshape(y(:,:,2),9,7);
Y(:,:,3) = reshape(y(:,:,3),9,7);
Y(:,:,4) = reshape(y(:,:,4),9,7);
Y(:,:,5) = reshape(y(:,:,5),9,7);
Y(:,:,6) = reshape(y(:,:,6),9,7);
Y(:,:,7) = reshape(y(:,:,7),9,7);
Y(:,:,8) = reshape(y(:,:,8),9,7);
Y(:,:,9) = reshape(y(:,:,9),9,7);
Y(:,:,10) = reshape(y(:,:,10),9,7);

%check recall

figure(1)
subplot(5,2,1),imagesc(S(:,:,1))
subplot(5,2,2),imagesc(Y(:,:,1))
subplot(5,2,3),imagesc(S(:,:,2))
subplot(5,2,4),imagesc(Y(:,:,2))
subplot(5,2,5),imagesc(S(:,:,3))
subplot(5,2,6),imagesc(Y(:,:,3))
subplot(5,2,7),imagesc(S(:,:,4))
subplot(5,2,8),imagesc(Y(:,:,4))
subplot(5,2,9),imagesc(S(:,:,5))
subplot(5,2,10),imagesc(Y(:,:,5))

figure(2)
subplot(5,2,1),imagesc(S(:,:,6))
subplot(5,2,2),imagesc(Y(:,:,6))
subplot(5,2,3),imagesc(S(:,:,7))
subplot(5,2,4),imagesc(Y(:,:,7))
subplot(5,2,5),imagesc(S(:,:,8))
subplot(5,2,6),imagesc(Y(:,:,8))
subplot(5,2,7),imagesc(S(:,:,9))
subplot(5,2,8),imagesc(Y(:,:,9))
subplot(5,2,9),imagesc(S(:,:,10))
subplot(5,2,10),imagesc(Y(:,:,10))


%Part B: When adding noise to the system, the more images present, the more
%resistant to noise it is. Although the outputs were not perfect recall for
%the 10 image set, the output given was very resistant to any noise.

%autoassociative NN using the Hebb rule
%Part C

%reusing same algorithm as part 1
inSize = size(S(:,:,1:10)); % matrix size of input

w = zeros(inSize(1),inSize(2),inSize(2),inSize(1)); %Matrix of weights for each input, each weight vector is a 7x9 mapped to the input indexes 9x7, 9x7x7x9
w_total = zeros(inSize(1),inSize(2),inSize(2),inSize(1)); % weights after adding up all input output wight matrices

for k = 1:inSize(3) % loop through all sets
    for i = 1:inSize(1)
        for j = 1:inSize(2)
            w(i,j,:,:) = S(i,j,k) * transpose(S(:,:,k)); %single weight matrix is S *T'
        end
    end
    w_total = w_total + w; % add all weight matrices being created
end

y = zeros(inSize);% set output of net to size of T

%train the network on third image until perfect recall
count = 0;
total = 0;
while(total ~= 63)
    for i = 1:inSize(1)
        for j = 1:inSize(2)
            w(i,j,:,:) = S(i,j,3) * transpose(S(:,:,3)); %single weight matrix is S *T'
        end
        w_total = w_total + w; % add all weight matrices being created
    end

    for i = 1:inSize(1) % loop through all weight vectors
        for j = 1:inSize(2)
            %disp(w_total(2,1,:,:))
            w_index = reshape(w_total(i,j,:,:),7,9);% reshape individual weight vector becasue matlab did not like it
            %disp(w_index)
            y(:,:,3) = transpose(w_index) * S(i,j,3); % y_in = W' * X
        end
    end

    for i = 1:inSize(1) % loop through all values of y_in and change to bipolar
        for j = 1:inSize(2)
            if (y(i,j,3) >= 0)
                y(i,j,3) = 1;
            else
                y(i,j,3) = -1;
            end
        end
    end
    count = count + 1;
    disp(count)
    %sum up equivalancy matrix of two matrices and if all values are the
    %same, total = 63
    total = sum(sum(y(:,:,3) == S(:,:,3)));
    disp(total)
end
%check recall
%left is input, right is output
figure(5)
subplot(5,2,1),imagesc(S(:,:,1))
subplot(5,2,2),imagesc(y(:,:,1))
subplot(5,2,3),imagesc(S(:,:,2))
subplot(5,2,4),imagesc(y(:,:,2))
subplot(5,2,5),imagesc(S(:,:,3))
subplot(5,2,6),imagesc(y(:,:,3))
subplot(5,2,7),imagesc(S(:,:,4))
subplot(5,2,8),imagesc(y(:,:,4))
subplot(5,2,9),imagesc(S(:,:,5))
subplot(5,2,10),imagesc(y(:,:,5))

figure(6)
subplot(5,2,1),imagesc(S(:,:,6))
subplot(5,2,2),imagesc(y(:,:,6))
subplot(5,2,3),imagesc(S(:,:,7))
subplot(5,2,4),imagesc(y(:,:,7))
subplot(5,2,5),imagesc(S(:,:,8))
subplot(5,2,6),imagesc(y(:,:,8))
subplot(5,2,7),imagesc(S(:,:,9))
subplot(5,2,8),imagesc(y(:,:,9))
subplot(5,2,9),imagesc(S(:,:,10))
subplot(5,2,10),imagesc(y(:,:,10))

%Part C: After running the algorithm for the third image 3 times, the weight matrix
%is weighted enouugh towards 3 that it will have perfect recall. This does
%improve the output of the system just for that image.