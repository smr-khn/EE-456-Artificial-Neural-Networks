%Self Organizing Map NN with 2D lattice driven by a 2D stimulus


%Write Up:
%Using the inputs and weights in the .mat file, I got a rough estimation of
%a SOM. There is some clumpin in parts but that may be fixed with some
%better fine tuning of the neighborhood radius. The ordering phase has a
%good spread of output nodes and has some node trying to chracterize every
%clust of inputs. The convergence phase has these output nodes getting
%close to the center of thesee clusters or some output nodes converge on a
%single input. Given more iterations on the convergence phase, a better SOM
%could be created but most of the converging errors has to do with these
%clusters of output nodes not being the closest to any single node thus
%never being updated.

clc,clear workspace, close all
load('testinput.mat')

%chosen values
learnrate = .8;
anneal = .99;
radius = .2;

%Create 200 (x,y) pairs with values between -1:1 for both 200 x 2 matrix
numIn = 200;
%input = rand(numIn,2)*2 -1;
%input is coming from testinput.mat

%initialize weights randomly betwwen [-.1,.1], 2 weights per output, 
%12x12 outputs = 144
numOut = 144;
%w = rand(numOut,2)*.2 -.1;
%weights loaded from test data

%output vector is 12 x 12 lattice of output neurons
output = zeros(numOut,1);

deltaw = 0;

%plot random inputs
figure(1)
scatter(input(:,1),input(:,2))
title('Input (x,y) plot and Weight vectors')

%plot random weight values
hold on
plot(w(:,1),w(:,2),'-x')
legend('Input','Weight')

i = 0; % index of input
numEpochs = 0;
k = 1; % total inputs run through

%ordering phase (150k-170k iterations)
while (numEpochs < 10)% stopping condition
    
    %iteration counter
    i = k - (ceil(k/numIn) - 1)*numIn;
    if(i==1)
        numEpochs = numEpochs + 1;
    end

    %find closest weight vector
    dist = zeros(numOut,1);
    for j = 1:numOut
        dist(j) = (input(i,1) - w(j,1)) ^ 2 + (input(i,2) - w(j,2)) ^ 2;
    end
    [value,closest] = min(dist); % closest holds index of weight closest to input
    
    closeNodes = zeros(numOut,1);
    for j = 1:numOut % find output nodes within neighborhood radius
        dist(j) = sqrt((w(closest,1) - w(j,1)) ^ 2 + (w(closest,2) - w(j,2)) ^ 2);
        if (dist(j) <= radius)
            closeNodes(j) = 1; % store 1 for weight index if the weight is within distance
        end
    end
    deltaw = learnrate * (input(i,:) - w(closest,:));
    for j = 1:numOut % update weights
        w(j,:) = w(j,:) + learnrate * (input(i,:) - w(closest,:)) * closeNodes(j); % w(new) = w(old) + alpha * (input - w(old)) * ifInNeighborhood(1 if node needs updated, 0 if not)
    end
    k = k+1; % increase counter
    divis = mod(numEpochs, 10);
    if(divis == 5 && learnrate > .01) % reduce learning rate exponentially but not less than .01
        learnrate = learnrate * anneal; % updates every 5 epochs
    end
    if(divis == 0 && radius > 0)% updates every 10 epochs
        radius = radius -.0005;
    end
    %figure(3)
    %scatter(w(:,1),w(:,2))
    %title('Weights after')
    disp(k)
end
%plot random inputs
figure(2)
scatter(input(:,1),input(:,2))
title('Input (x,y) plot and Weight vectors after ordering (10 epochs)')

%plot random weight values
hold on
plot(w(:,1),w(:,2),'-x')
legend('Input','Weight')

%convergence phase (5x iterations than the ordering phase) iterations
learnrate = .01; % set eta to a small constant and do not update it, radius should already be 0 from ordering phase
%radius = .01;
while (numEpochs < 100)% stopping condition
    
    %iteration counter
    i = k - (ceil(k/numIn) - 1)*numIn;
    if(i==1)
        numEpochs = numEpochs + 1;
    end

    %find closest weight vector
    dist = zeros(numOut,1);
    for j = 1:numOut
        dist(j) = (input(i,1) - w(j,1)) ^ 2 + (input(i,2) - w(j,2)) ^ 2;
    end
    [value,closest] = min(dist); % closest holds index of weight closest to input
    
    closeNodes = zeros(numOut,1);
    for j = 1:numOut % find output nodes within neighborhood radius
        dist(j) = sqrt((w(closest,1) - w(j,1)) ^ 2 + (w(closest,2) - w(j,2)) ^ 2);
        if (dist(j) <= radius)
            closeNodes(j) = 1; % store 1 for weight index if the weight is within distance
        end
    end
    deltaw = learnrate * (input(i,:) - w(closest,:));
    for j = 1:numOut % update weights
        w(j,:) = w(j,:) + deltaw * closeNodes(j); % w(new) = w(old) + alpha * (input - w(old)) * ifInNeighborhood(1 if node needs updated, 0 if not)
    end
    k = k+1; % increase counter
    disp(k)
end

figure(3)
scatter(input(:,1),input(:,2))
title('Input (x,y) plot and Weight vectors after convergence (90 epochs)')

%plot random weight values
hold on
plot(w(:,1),w(:,2),'-x')
legend('Input','Weight')
