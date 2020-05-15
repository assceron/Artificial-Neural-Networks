close all
clear all

beta = 2;
N = 200;
sizeOfPatterns_1 = 5;
sizeOfPatterns_2 = 40;
T = 10^5;
all_m1_1 = zeros(1,100);
all_m1_2 = zeros(1,100);


for i = 1:100
    all_m1_1(i) = compute_m1_wii_zero(sizeOfPatterns_1, N, beta, T);
end
mean_1 = mean(all_m1_1);

for i = 1:100
    all_m1_2(i) = compute_m1_wii_zero(sizeOfPatterns_2, N, beta, T);
end
mean_2 = mean(all_m1_2);

function m1 = compute_m1_wii_zero(sizeOfPatterns, N, beta, T)
        %generation of patterns
        patterns = generatePatterns(sizeOfPatterns, N);
        %initialization of weights
        weights = zeros(N,N);
        weights = initializeWeights_wii_zero(weights,N,patterns);
        %feeding the network
        toFeed = patterns(1,:);
        neurons = toFeed;
        partial_m1 = 0;
        %network dynamics
        for i=1:T
            toUpdate = randi([1,N]);
            chosenWeights = weights(toUpdate,:);
            chosenNeuron = localField(chosenWeights,neurons);
            neurons(toUpdate) = neuronUpdate(chosenNeuron,beta);
            partial_m1 = partial_m1 + 1/N * (compute_sum(neurons, toFeed, N));
        end
        m1 = 1/T * partial_m1;
end

function sum=compute_sum(neurons, toFeed,N)
    sum = 0;
    for i=1:N
        sum = sum + neurons(i)*toFeed(i);
    end
end

function weights=initializeWeights_wii_zero(weights,N, patterns)
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

function sign = neuronUpdate(number,beta)
    exponent = -2*beta*number;
    probabilityValue = 1/(1+(exp(exponent)));
    x = rand;
    if x < probabilityValue
        sign = 1;
    else
        sign = -1;
    end
end

%local field computation
function b=localField(weights,neurons)
    b=weights*transpose(neurons);
end

