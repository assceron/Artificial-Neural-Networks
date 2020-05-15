%%
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadMNIST(1);

%%
%settings
outputsize = 10;
inputsize = 784;
hiddenNeurons = 30;
learning_rate = 0.3;
batchsize = 10;
numberOfEpochs = 30;

[hiddenThresholds, hiddenWeights, outputThresholds, outputWeights, C_T_net2, C_V_net2,epochForMinimumValid_net2, minimumValid_net2] = train(tTrain, xTrain, xValid, tValid, xTest, tTest, learning_rate, inputsize, outputsize, batchsize, numberOfEpochs, hiddenNeurons);
%%

%figure(), semilogy(C_V), hold, semilogy(C_T), legend('error valid', 'error train')
%%
[xTrain, meanToShift] = shiftTrain(xTrain);
xTest = shift(xTest, meanToShift);
xValid = shift(xValid, meanToShift);
 for u = 1:size(xTest,2)
      local_field_V1 = hiddenWeights*xTest(:,u) - hiddenThresholds;
      V1 = sigmoid(local_field_V1);
      local_field_output = outputWeights*V1 - outputThresholds;
      outputTest = sigmoid(local_field_output);
      processedOutputTest = postProcess(outputTest);
      C_test(u) = sum(abs(tTest(:,u)-processedOutputTest));
 end
 C_test_final_net2 = 0.5*mean(C_test);
 
 for u = 1:size(xTest,2)
      local_field_V1 = hiddenWeights*xTrain(:,u) - hiddenThresholds;
      V1 = sigmoid(local_field_V1);
      local_field_output = outputWeights*V1 - outputThresholds;
      outputTrain = sigmoid(local_field_output);
      processedOutputTrain = postProcess(outputTrain);
      C_train(u) = sum(abs(tTrain(:,u)-processedOutputTrain));
 end
 C_train_final_net2 = 0.5*mean(C_train);
 
 for u = 1:size(xValid,2)
      local_field_V1 = hiddenWeights*xValid(:,u) - hiddenThresholds;
      V1 = sigmoid(local_field_V1);
      local_field_output = outputWeights*V1 - outputThresholds;
      outputValid = sigmoid(local_field_output);
      processedOutputValid = postProcess(outputValid);
      C_valid(u) = sum(abs(tValid(:,u)-processedOutputValid));
 end
 C_valid_final_net2 = 0.5*mean(C_valid);
%%
function [hiddenThresholds, hiddenWeights, outputThresholds, outputWeights, C_T, C_V, epochForMinimumValid, minimumValid] = train(tTrain, xTrain, xValid, tValid, xTest, tTest, learning_rate, inputsize, outputsize, batchsize, numberOfEpochs, hiddenNeurons)
       
    %initialization of weights and thresholds
    numberOfWeightsFeedingOutput = hiddenNeurons;
    numberOfWeightsFeedingHidden = inputsize;
    outputWeights = initializeWeights(outputsize,hiddenNeurons,numberOfWeightsFeedingOutput);
    hiddenWeights = initializeWeights(hiddenNeurons,inputsize,numberOfWeightsFeedingHidden);
    outputThresholds = zeros(outputsize,1);
    hiddenThresholds = zeros(hiddenNeurons,1);
    
    %shift of the sets
    [xTrain, meanToShift] = shiftTrain(xTrain);
	xValid = shift(xValid, meanToShift);
    
    %training start
    for t=1:numberOfEpochs
        t
        
        %classification error computation
        for u = 1:size(xTrain,2)
            X=xTrain(:,u);
            local_field_V1 = hiddenWeights*X - hiddenThresholds;
            V1 = sigmoid(local_field_V1);
            local_field_output = outputWeights*V1 - outputThresholds;
            outputTrain = sigmoid(local_field_output);
            processedOutputTrain = postProcess(outputTrain);
            C_train(u) = sum(abs(tTrain(:,u)-processedOutputTrain));
        end
        
        for u = 1:size(xValid,2)
            X=xValid(:,u);
            local_field_V1 = hiddenWeights*X - hiddenThresholds;
            V1 = sigmoid(local_field_V1);
            local_field_output = outputWeights*V1 - outputThresholds;
            outputValid = sigmoid(local_field_output);
            processedOutputValid = postProcess(outputValid);
            C_valid(u) = sum(abs(tValid(:,u)-processedOutputValid));
        end

        C_T(t) = 0.5*mean(C_train);
        C_V(t) = 0.5*mean(C_valid);
        
        %save weights and thresholds in case of best result
        if (isequal(C_V(t),min(C_V)))
            definitiveHiddenWeights = hiddenWeights;
            definitiveHiddenThresholds = hiddenThresholds;
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
    
            delta_hiddenWeights = zeros(size(hiddenWeights));
            delta_hiddenThresholds = zeros(size(hiddenThresholds));
        
            %computation of the output of the batch
            for u=1:size(xBatch,2)
                %feed forward
                X = xBatch(:,u);
                local_field_V1 = hiddenWeights*X - hiddenThresholds;
                V1 = sigmoid(local_field_V1);
                local_field_output = outputWeights*V1 - outputThresholds;
                V2 = sigmoid(local_field_output);
                
                outputError = (tBatch(:,u)-V2) .* sigmoid(local_field_output) .* (1-sigmoid(local_field_output));
                hiddenError = outputWeights' * outputError .* sigmoid(local_field_V1) .* (1-sigmoid(local_field_V1));

                delta_outputWeights = delta_outputWeights + learning_rate * outputError * V1';
                delta_hiddenWeights = delta_hiddenWeights + learning_rate * hiddenError * X';
                
                delta_outputThresholds = delta_outputThresholds - learning_rate * outputError;
                delta_hiddenThresholds = delta_hiddenThresholds - learning_rate * hiddenError;
            end

        hiddenWeights = hiddenWeights + delta_hiddenWeights;
        hiddenThresholds = hiddenThresholds + delta_hiddenThresholds;
            
        outputWeights = outputWeights + delta_outputWeights;
        outputThresholds = outputThresholds + delta_outputThresholds;
        
        p=p+1;
        end
    end
    
        outputWeights = definitiveOutputWeights;
        hiddenWeights = definitiveHiddenWeights;
        
        outputThresholds = definitiveOutputThresholds;
        hiddenThresholds = definitiveHiddenThresholds;
          
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
