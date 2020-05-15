clear all
close all

functions = csvread('input_data_numeric.csv');
functions(:,1) = [];
all_targets =  [[-1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
    [1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1],
    [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1],
    [1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1],
    [1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
    [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1]];
outputsize = 16;
learning_rate = 0.02;
results = zeros(1,6);

for i=1:6
targets = all_targets(i,:);
[threshold, weights] = train(targets, functions, learning_rate, outputsize);
outputs = sign(computeOutput(weights, functions, threshold, outputsize)); 

C = 0.5 * sum(abs(targets-sign(outputs)));

if (not(C==0))
    for counter=1:10
        counter
        [threshold, weights] = train(targets, functions, learning_rate, outputsize);
        outputs = (computeOutput(weights, functions, threshold, outputsize));
        if (C==0)
            break
        end
    end
end

if (C==0)
    results(i)=1;
end

end

function [threshold, weights] = train(targets, functions, learning_rate, outputsize)
    weights = initializeRandomWeights;
    threshold =  -1 + (1-(-1)).*rand(1,1);
    delta_weights = zeros(1,4);
    for t=1:10^5
        j = randi([1, outputsize]);
        randomPattern = functions(j,:);
        outputs = computeOutput(weights, functions, threshold, outputsize);
        for i=1:4
            delta_weights(i) = 0.5 * (targets(j)-outputs(j)) * (1-(tanh(0.5*((sum(weights.*randomPattern))-threshold))^2)) * randomPattern(i);
        end
        delta_threshold = -0.5 * (targets(j)-outputs(j)) * (1-(tanh(0.5*((sum(weights.*randomPattern))-threshold))^2));
        weights = weights + learning_rate * delta_weights;
        threshold = threshold + learning_rate * delta_threshold;
    end
end

function outputs = computeOutput(weights, functions, threshold, outputsize)
    outputs = zeros(1,outputsize);
    for i=1:outputsize
        outputs(i) = tanh(0.5 * ((sum(weights.*functions(i,:)) - threshold)));
    end
end

function weights = initializeRandomWeights
   weights = -0.2 + (0.2-(-0.2)).*rand(1,4);
end
