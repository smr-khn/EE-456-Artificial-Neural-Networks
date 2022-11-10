% heteroassociative neural net using the Hebb rule
%Part A
load('S_patterns_HW3_P2.mat')
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
