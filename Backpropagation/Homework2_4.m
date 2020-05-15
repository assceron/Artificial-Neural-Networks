close all

training = csvread('training_set.csv');
validation = csvread('validation_set.csv');
learningRate = 0.02;

attempts = 1;
table = zeros(attempts,5); 
for k=1:100
    learningRate = 0.017 + (0.02-(-0.017)).*rand(1,1);
    M1 = randi([39,43],1,1);
    M2 = randi([20,23],1,1);
    updates = 100000; 
   [inputWeights, hiddenWeights, outputWeights, firstThresholds, secondThresholds, outputThreshold, H] = train(M1,M2,learningRate,training, updates);
    output = computeOutput(inputWeights, hiddenWeights, outputWeights, firstThresholds, secondThresholds, outputThreshold, validation, M1, M2, 5000);
    signOutput = sign(output);
    C = (1/(2*5000)) * sum(abs(validation(:,3)-signOutput'));
    table(k,1) = M1;
    table(k,2) = M2;
    table(k,3) = learningRate;
    table(k,4) = updates;
    table(k,5) = C;
    if (C<0.12)
        break
    end
end


csvwrite('w1.csv',inputWeights);
csvwrite('w2.csv',hiddenWeights);
csvwrite('w3.csv',outputWeights);
csvwrite('t1.csv',firstThresholds);
csvwrite('t2.csv',secondThresholds);
csvwrite('t3.csv',outputThreshold);


function [inputWeights, hiddenWeights, outputWeights, firstThresholds, secondThresholds, outputThreshold, H] = train(M1,M2, learningRate, training, updates)
inputWeights = initializeRandomWeights(M1, 2);
hiddenWeights = initializeRandomWeights(M2, M1);
outputWeights = initializeRandomWeights(M2, 1);
firstThresholds =  normrnd(0,1,1,M1);
secondThresholds =  normrnd(0,1,1,M2);
outputThreshold =  normrnd(0,1,1,1);
H = zeros(1,updates);
for t = 1:updates
    %pick random pattern
    p = randi([1, 10^4]);
    randomPattern = training(p,:);
    randomPattern_x(1) = randomPattern(1);
    randomPattern_x(2) = randomPattern(2);
    %feed forward
    firstLayer = zeros(1, M1);
    for j = 1:M1
        firstLayer(j) = tanh(sum(inputWeights(j,:).*randomPattern_x)-firstThresholds(j));
    end
    secondLayer = zeros(1, M2);
    for i = 1:M2
        secondLayer(i) = tanh(sum(hiddenWeights(i,:).*firstLayer)-secondThresholds(i));
    end
    target = training(p,3);
    %weights update backward
    delta_3 = (target - tanh((sum(outputWeights.*secondLayer') - outputThreshold))) * (1-(tanh(sum(outputWeights.*secondLayer')-outputThreshold)^2));
    delta_3_weights = zeros(1,M2);
    for i=1:M2
        delta_3_weights(i) = learningRate * delta_3 * secondLayer(i);
    end
    delta_2_weights = zeros(M2,M1);
    delta_2 = zeros(1,M2);
    for i=1:M2
        delta_2(i) = delta_3 * outputWeights(i) * (1-(tanh(sum(hiddenWeights(i,:).*firstLayer)-secondThresholds(i)))^2);
        for j=1:M1
            delta_2_weights(i,j) = learningRate * delta_2(i) * firstLayer(j);
        end
    end
    delta_1_weights = zeros(M1,2);
    delta_1 = zeros(1,M1);
    for i=1:M1
        delta_1(i) = sum(delta_2.*hiddenWeights(:,i)')*(1-(tanh(sum(inputWeights(i,:).*randomPattern_x)-firstThresholds(i)))^2);
        for j=1:2
            delta_1_weights(i,j) = learningRate * delta_1(i) * randomPattern_x(j);
        end
    end

    delta_3_threshold = - learningRate * delta_3;
    delta_2_threshold = zeros(1,M2);
    for i=1:M2
        delta_2_threshold(i) = - learningRate * delta_2(i);
    end
    delta_1_threshold = zeros(1,M1);
    for i=1:M1
        delta_1_threshold(i) = - learningRate * delta_1(i);
    end
    
    outputWeights = outputWeights + delta_3_weights';
    hiddenWeights = hiddenWeights + delta_2_weights;
    inputWeights = inputWeights + delta_1_weights;
    
    outputThreshold = outputThreshold + delta_3_threshold;
    secondThresholds = secondThresholds + delta_2_threshold;
    firstThresholds = firstThresholds + delta_1_threshold;

%     output = computeOutput(inputWeights, hiddenWeights, outputWeights, firstThresholds, secondThresholds, outputThreshold, training, M1, M2, 10000);
%     H(t) = 0.5 * sum((training(:,3)-output').^2);
    end
end

function outputs = computeOutput(inputWeights, hiddenWeights, outputWeights, firstThresholds, secondThresholds, outputThreshold, validation, M1, M2, size)
    outputs = zeros(1,size);
    for u=1:size
        randomPattern = validation(u,:);
        randomPattern_x(1) = randomPattern(1);
        randomPattern_x(2) = randomPattern(2);
        firstLayer = zeros(1,M1);
        for j = 1:M1
            firstLayer(j) = tanh(sum(inputWeights(j,:).*randomPattern_x)-firstThresholds(j));
        end
        secondLayer = zeros(1,M2);
        for i = 1:M2
            secondLayer(i) = tanh(sum(hiddenWeights(i,:).*firstLayer)-secondThresholds(i));
        end
        outputs(u) = tanh(sum(outputWeights.*secondLayer')-outputThreshold);
    end
end

function weights = initializeRandomWeights(rows, columns)
   weights = normrnd(0,1,[rows,columns]);
end