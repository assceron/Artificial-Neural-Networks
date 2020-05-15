
%%
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadMNIST(1);
%%
%settings
outputsize = 10;
inputsize = 784;
learning_rate = 0.3;
batchsize = 10;
numberOfEpochs = 30;
%%
%training
[thresholds, weights, C_T_net1, C_V_net1, meanToShift, epochForMinimumValid_net1, minimumValid_net1] = train(tTrain, xTrain, xValid, tValid, learning_rate, inputsize, outputsize, batchsize, numberOfEpochs);
%plot of the results
%figure(), semilogy(C_V), hold, semilogy(C_T), legend('error valid', 'error train')
%%
[xTrain, meanToShift] = shiftTrain(xTrain);
xTest = shift(xTest, meanToShift);
xValid = shift(xValid, meanToShift);
C_test=zeros(size(xTest,2),1);
C_train=zeros(size(xTrain,2),1);
C_valid=zeros(size(xValid,2),1);

for u = 1:size(xTest,2)
    local_field = weights*xTest(:,u) - thresholds;
    outputTest = sigmoid(local_field);
    processedOutputTest = postProcess(outputTest);
    C_test(u) = sum(abs(tTest(:,u)-processedOutputTest));
end
C_test_final_net1 = 0.5*mean(C_test);

for u = 1:size(xTrain,2)
    local_field = weights*xTrain(:,u) - thresholds;
    outputTrain = sigmoid(local_field);
    processedOutputTrain = postProcess(outputTrain);
    C_train(u) = sum(abs(tTrain(:,u)-processedOutputTrain));
end
C_train_final_net1 = 0.5*mean(C_train);

for u = 1:size(xValid,2)
    local_field = weights*xValid(:,u) - thresholds;
    outputValid = sigmoid(local_field);
    processedOutputValid = postProcess(outputValid);
    C_valid(u) = sum(abs(tValid(:,u)-processedOutputValid));
end
C_valid_final_net1 = 0.5*mean(C_valid);

%%
 function [thresholds, weights, C_T, C_V, meanToShift, epochForMinimumValid, minimumValid] = train(tTrain, xTrain, xValid, tValid, learning_rate, inputsize, outputsize, batchsize, numberOfEpochs)
       
    %initialization of weights and thresholds
    numbersOfWeightsFeedingOutput = inputsize;
    weights = initializeWeights(outputsize,inputsize,numbersOfWeightsFeedingOutput);
    thresholds = zeros(outputsize,1);
    
    delta_weights = zeros(size(weights));
    delta_thresholds = zeros(size(thresholds));
    
    %shift of the sets
    [xTrain, meanToShift] = shiftTrain(xTrain);
	xValid = shift(xValid, meanToShift);
    
    %training start
    for t=1:numberOfEpochs
        t
        
        %classification error computation
        for u = 1:size(xTrain,2)
            local_field = weights*xTrain(:,u) - thresholds;
            outputTrain = sigmoid(local_field);
            processedOutputTrain = postProcess(outputTrain);
            C_train(u) = sum(abs(tTrain(:,u)-processedOutputTrain));
        end
        
        for u = 1:size(xValid,2)
            local_field = weights*xValid(:,u) - thresholds;
            outputValid = sigmoid(local_field);
            processedOutputValid = postProcess(outputValid);
            C_valid(u) = sum(abs(tValid(:,u)-processedOutputValid));
        end

        C_T(t) = 0.5*mean(C_train);
        C_V(t) = 0.5*mean(C_valid);
        
        %save weights and thresholds in case of best result
        if (isequal(C_V(t),min(C_V)))
            definitiveWeights = weights;
            definitiveThresholds = thresholds;
            epochForMinimumValid = t;
            minimumValid = min(C_V);
        end
        
        xFull = [xTrain; tTrain];
        xFull(:,randperm(size(xFull,2)));

        p=1;
        while (not((p+9)==50000))
            batch = xFull(:,p:(p+9));
        
            xBatch = batch(1:784,:);
            tBatch = batch(785:794,:);
            
            delta_weights = zeros(size(weights));
            delta_thresholds = zeros(size(thresholds));
            %computation of the output of the batch
            for u=1:size(xBatch,2)
                local_field = weights*xBatch(:,u) - thresholds;
                outputBatch = sigmoid(local_field);
                
                outputError = (tBatch(:,u)-outputBatch) .* sigmoid(local_field) .* (1-sigmoid(local_field));
                
                delta_weights = delta_weights + learning_rate * outputError * xBatch(:,u)';
                delta_thresholds = delta_thresholds - learning_rate * outputError;
            end

            weights = weights + delta_weights;
            thresholds = thresholds + delta_thresholds;
            
            p=p+10;
        end
    end
    
    weights = definitiveWeights;
    thresholds = definitiveThresholds;
    
%     if (t==30)
%         for u=1:size(xTrain,2)
%             local_field = weights*xTrain(:,u) - thresholds;
%             totalOutputTrain(:,u) = sigmoid(local_field);
%             processedOutputTrain(:,u) = postProcess(totalOutputTrain(:,u));
%         end
%     end
    
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


