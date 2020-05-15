clear alll
close all

sizeOfPatterns = 12;
N = 100;
numbersOfPatterns = [12,20,40,60,80,100];
p1 = [0,0,0,0,0,0];
p2 = [0,0,0,0,0,0];

for i=1:length(numbersOfPatterns)
    patterns, weights = computeError(numbersOfPatterns(i),100);
end

function [patterns,weights] = computeError(sizeOfPatterns, N)
    cumulativeError=0;
    for i=1:10^5
        i
        %generation of patterns
        fprintf("I generate patterns")
        patterns = generatePatterns(sizeOfPatterns, N);
        %initialization of weights
        weights = zeros(100,100);
        weights = initializeWeights(weights,N, sizeOfPatterns,patterns);
        %feeding the network
        idx = randi([1,sizeOfPatterns]);
        toFeed = patterns(idx,:);
        fprintf("I'm going to feed the network")
        neurons = toFeed;
        %random choosing neuron to update
        toUpdate = randi([1,N]);
        %update of the neuron
        chosenNeuron = neurons(toUpdate);
        chosenWeights = weights(toUpdate,:);
        chosenNeuron = localField(chosenWeights,neurons);
        neurons(toUpdate) = mySign(chosenNeuron);
        if (~isequal(neurons,toFeed))
            fprintf("I am here")
            cumulativeError = cumulativeError+1;
        end
    end
    probability = cumulativeError/100000;
end

function probability = computeError_wii_zero(sizeOfPatterns, N)
    cumulativeError=0;
    for i=1:1
        i
        %generation of patterns
        fprintf("I generate patterns")
        patterns = generatePatterns(sizeOfPatterns, N);
        %initialization of weights
        weights = zeros(100,100);
        weights = initializeWeights_wii_zero(weights,N, sizeOfPatterns,patterns);
        %feeding the network
        idx = randi([1,sizeOfPatterns]);
        toFeed = patterns(idx,:);
        fprintf("I'm going to feed the network")
        neurons = toFeed;
        %random choosing neuron to update
        toUpdate = randi([1,N]);
        %update of the neuron
        chosenNeuron = neurons(toUpdate);
        chosenWeights = weights(toUpdate,:);
        chosenNeuron = localField(chosenWeights,neurons);
        neurons(toUpdate) = mySign(chosenNeuron);
        if (~isequal(neurons,toFeed))
            fprintf("I am here")
            cumulativeError = cumulativeError+1;
        end
    end
    probability = cumulativeError/100000;
end

%use the Hebb's rule to store the patterns
function weights=initializeWeights(weights,N,sizeOfPatterns,patterns)
    for i=1:N
        for j=1:N
            weightValue = 0;
            weightValue = sum(patterns(:,i).*patterns(:,j));
            weights(i,j) = 1/N * weightValue;
        end
    end
end

function weights=initializeWeights_wii_zero(weights,N,sizeOfPatterns,patterns)
    for i=1:N
        for j=1:N
            weightValue = 0;
            weightValue = sum(patterns(:,i).*patterns(:,j));
            weights(i,j) = 1/N * weightValue;
            if (i==j)
                weights(i,j) = 0;
            end
        end
    end
end

function patterns = generatePatterns(sizeOfPatterns, N)
    patterns = zeros(sizeOfPatterns,N);
    possibleValues = [-1, 1];
    for i=1:sizeOfPatterns
        for j=1:N
            randomIndex = randi(length(possibleValues), 1);
            patterns(i,j) = possibleValues(randomIndex);
        end
    end
end

function sign=mySign(number)
    if number >= 0
        sign = 1;
    else
        sign = -1;
    end
end

%local field computation
function b=localField(weights,neurons)
    b=weights*transpose(neurons);
end