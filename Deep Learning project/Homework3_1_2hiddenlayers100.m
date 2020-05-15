%%
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadMNIST(1);

%%
%settings
outputsize = 10;
inputsize = 784;
hiddenNeurons = 100;
learning_rate = 0.3;
batchsize = 10;
numberOfEpochs = 30;

[hidden_1_Thresholds, hidden_1_Weights, hidden_2_Thresholds, hidden_2_Weights,outputThresholds, outputWeights, C_T_net4, C_V_net4,epochForMinimumValid_net4, minimumValid_net4] = train(tTrain, xTrain, xValid, tValid, xTest, tTest, learning_rate, inputsize, outputsize, batchsize, numberOfEpochs, hiddenNeurons);

%%
[xTrain, meanToShift] = shiftTrain(xTrain);
xTest = shift(xTest, meanToShift);
xValid = shift(xValid, meanToShift);
for u = 1:size(xTest,2)
    X=xTest(:,u);
    local_field_V1 = hidden_1_Weights*X - hidden_1_Thresholds;
    V1 = sigmoid(local_field_V1);
    local_field_V2 = hidden_2_Weights*V1 - hidden_2_Thresholds;
    V2 = sigmoid(local_field_V2);
    local_field_output = outputWeights*V2 - outputThresholds;
    outputTest = sigmoid(local_field_output);
    processedOutputTest = postProcess(outputTest);
    C_test(u) = sum(abs(tTest(:,u)-processedOutputTest));
end
C_test_final_net4 = 0.5*mean(C_test);
 
for u = 1:size(xTrain,2)
    X=xTrain(:,u);
    local_field_V1 = hidden_1_Weights*X - hidden_1_Thresholds;
    V1 = sigmoid(local_field_V1);
    local_field_V2 = hidden_2_Weights*V1 - hidden_2_Thresholds;
    V2 = sigmoid(local_field_V2);
    local_field_output = outputWeights*V2 - outputThresholds;
    outputTrain = sigmoid(local_field_output);
    processedOutputTrain = postProcess(outputTrain);
    C_train(u) = sum(abs(tTrain(:,u)-processedOutputTrain));
end
C_train_final_net4 = 0.5*mean(C_train);
 
for u = 1:size(xValid,2)
    X=xValid(:,u);
    local_field_V1 = hidden_1_Weights*X - hidden_1_Thresholds;
    V1 = sigmoid(local_field_V1);
    local_field_V2 = hidden_2_Weights*V1 - hidden_2_Thresholds;
    V2 = sigmoid(local_field_V2);
    local_field_output = outputWeights*V2 - outputThresholds;
    outputValid = sigmoid(local_field_output);
    processedOutputValid = postProcess(outputValid);
    C_valid(u) = sum(abs(tValid(:,u)-processedOutputValid));
end
C_valid_final_net4 = 0.5*mean(C_valid);
%%
function [hidden_1_Thresholds, hidden_1_Weights, hidden_2_Thresholds, hidden_2_Weights, outputThresholds, outputWeights, C_T, C_V, epochForMinimumValid, minimumValid] = train(tTrain, xTrain, xValid, tValid, xTest, tTest, learning_rate, inputsize, outputsize, batchsize, numberOfEpochs, hiddenNeurons)
       
    %initialization of weights and thresholds
    numberOfWeightsFeedingOutput = hiddenNeurons;
    numberOfWeightsFeedingHidden = inputsize;
    outputWeights = initializeWeights(outputsize,hiddenNeurons,numberOfWeightsFeedingOutput);
    hidden_1_Weights = initializeWeights(hiddenNeurons,inputsize,numberOfWeightsFeedingHidden);
    hidden_2_Weights = initializeWeights(hiddenNeurons,hiddenNeurons,numberOfWeightsFeedingOutput);
    outputThresholds = zeros(outputsize,1);
    hidden_1_Thresholds = zeros(hiddenNeurons,1);
    hidden_2_Thresholds = zeros(hiddenNeurons,1);
    
    %shift of the sets
    [xTrain, meanToShift] = shiftTrain(xTrain);
	xValid = shift(xValid, meanToShift);
    
    %training start
    for t=1:numberOfEpochs
        t
        
        %classification error computation
        for u = 1:size(xTrain,2)
            X=xTrain(:,u);
            local_field_V1 = hidden_1_Weights*X - hidden_1_Thresholds;
            V1 = sigmoid(local_field_V1);
            local_field_V2 = hidden_2_Weights*V1 - hidden_2_Thresholds;
            V2 = sigmoid(local_field_V2);
            local_field_output = outputWeights*V2 - outputThresholds;
            outputTrain = sigmoid(local_field_output);
            processedOutputTrain = postProcess(outputTrain);
            C_train(u) = sum(abs(tTrain(:,u)-processedOutputTrain));
        end
        
       for u = 1:size(xValid,2)
            X=xValid(:,u);
            local_field_V1 = hidden_1_Weights*X - hidden_1_Thresholds;
            V1 = sigmoid(local_field_V1);
            local_field_V2 = hidden_2_Weights*V1 - hidden_2_Thresholds;
            V2 = sigmoid(local_field_V2);
            local_field_output = outputWeights*V2 - outputThresholds;
            outputValid = sigmoid(local_field_output);
            processedOutputValid = postProcess(outputValid);
            C_valid(u) = sum(abs(tValid(:,u)-processedOutputValid));
        end

        C_T(t) = 0.5*mean(C_train);
        C_V(t) = 0.5*mean(C_valid);
        
        %save weights and thresholds in case of best result
        if (isequal(C_V(t),min(C_V)))
            definitiveHidden_1_Weights = hidden_1_Weights;
            definitiveHidden_1_Thresholds = hidden_1_Thresholds;
            definitiveHidden_2_Weights = hidden_2_Weights;
            definitiveHidden_2_Thresholds = hidden_2_Thresholds;
            definitiveOutputWeights = outputWeights;
            definitiveOutputThresholds = outputThresholds;
            epochForMinimumValid = t;
            minimumValid = min(C_V);
        end
        
        xFull = [xTrain; tTrain];
        xFull(:,randperm(size(xFull,2)));
        %shuffle of Training set
        p=1;
        while (p~=5000)
            ncol = 10;
            x = randi(size(xFull,2),1,ncol);
            batch = xFull(:,x);
            
            xBatch = batch(1:784,:);
            tBatch = batch(785:794,:);
            
            delta_outputWeights = zeros(size(outputWeights));
            delta_outputThresholds = zeros(size(outputThresholds));
    
            delta_hidden_1_Weights = zeros(size(hidden_1_Weights));
            delta_hidden_1_Thresholds = zeros(size(hidden_1_Thresholds));
            
            delta_hidden_2_Weights = zeros(size(hidden_2_Weights));
            delta_hidden_2_Thresholds = zeros(size(hidden_2_Thresholds));
        
            %computation of the output of the batch
            for u=1:size(xBatch,2)
                %feed forward
                X = xBatch(:,u);
                local_field_V1 = hidden_1_Weights*X - hidden_1_Thresholds;
                V1 = sigmoid(local_field_V1);
                local_field_V2 = hidden_2_Weights*V1 - hidden_2_Thresholds;
                V2 = sigmoid(local_field_V2);
                local_field_output = outputWeights*V2 - outputThresholds;
                output = sigmoid(local_field_output);
                
                outputError = (tBatch(:,u)-output) .* sigmoid(local_field_output) .* (1-sigmoid(local_field_output));
                hidden_2_Error = outputWeights' * outputError .* sigmoid(local_field_V2) .* (1-sigmoid(local_field_V2));
                hidden_1_Error = hidden_2_Weights' * hidden_2_Error .* sigmoid(local_field_V1) .* (1-sigmoid(local_field_V1));

                delta_outputWeights = delta_outputWeights + learning_rate * outputError * V2';
                delta_hidden_2_Weights = delta_hidden_2_Weights + learning_rate * hidden_2_Error * V1';
                delta_hidden_1_Weights = delta_hidden_1_Weights + learning_rate * hidden_1_Error * X';


                delta_outputThresholds = delta_outputThresholds - learning_rate * outputError;
                delta_hidden_1_Thresholds = delta_hidden_1_Thresholds - learning_rate * hidden_1_Error;
                delta_hidden_2_Thresholds = delta_hidden_2_Thresholds - learning_rate * hidden_2_Error;

            end

        hidden_1_Weights = hidden_1_Weights + delta_hidden_1_Weights;
        hidden_1_Thresholds = hidden_1_Thresholds + delta_hidden_1_Thresholds;
        
        hidden_2_Weights = hidden_2_Weights + delta_hidden_2_Weights;
        hidden_2_Thresholds = hidden_2_Thresholds + delta_hidden_2_Thresholds;
            
        outputWeights = outputWeights + delta_outputWeights;
        outputThresholds = outputThresholds + delta_outputThresholds;
        
        p=p+1;
        end
    end
    
        outputWeights = definitiveOutputWeights;
        hidden_1_Weights = definitiveHidden_1_Weights;
        hidden_2_Weights = definitiveHidden_2_Weights;

        outputThresholds = definitiveOutputThresholds;
        hidden_1_Thresholds = definitiveHidden_1_Thresholds;
        hidden_2_Thresholds = definitiveHidden_2_Thresholds;

 end
 
 function processedOutput = postProcess(output)
    [~, winning] = max(output);
    processedOutput = zeros(size(output,1),1);
    processedOutput(winning)=1;
 end

function xSet = shift(xSet, meanToShift)
    matrixToShift = zeros(size(xSet));
    for i=1:size(xSet,2)
       matrixToShift(:,i) = meanToShift;
    end
    xSet = xSet - matrixToShift;
end

function [xTrain, meanToShift] = shiftTrain(xTrain)
    meanToShift = zeros(size(xTrain,1),1);
    for i=1:size(xTrain,1)
        meanToShift(i) = mean(xTrain(i,:));
    end
    matrixToShift = zeros(size(xTrain));
    for i=1:size(xTrain,2)
       matrixToShift(:,i) = meanToShift;
    end
    xTrain = xTrain - matrixToShift;
end

function weights = initializeWeights(rows, columns,numbersOfWeightsFeedingLevel)
   weights = normrnd(0,1/sqrt(numbersOfWeightsFeedingLevel),[rows,columns]);
end
