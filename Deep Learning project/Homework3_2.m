%%
clear all
close all
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadMNIST(2);

%%
%%
%settings
outputsize = 10;
inputsize = 784;
hiddenNeurons = 30;
learning_rate = 3*10^(-3);
batchsize = 10;
numberOfEpochs = 50;

%%
[hidden_1_Thresholds, hidden_1_Weights, hidden_2_Weights, hidden_2_Thresholds,hidden_3_Weights, hidden_3_Thresholds,hidden_4_Weights, hidden_4_Thresholds, outputThresholds, outputWeights, u5, u4, u3, u2, u1, energy] = train(tTrain, xTrain, xValid, tValid, xTest, tTest, learning_rate, inputsize, outputsize, batchsize, numberOfEpochs, hiddenNeurons);
%%
semilogy(u5), hold on, semilogy(u4),hold on, semilogy(u3), hold on, semilogy(u2), hold on, semilogy(u1), hold off, legend('|u^{(5)}|','|u^{(4)}|','|u^{(3)}|','|u^{(2)}|','|u^{(1)}|')
title('Learning speeds for Homework 3.2')
xlabel('Epochs') 
ylabel('Learning speeds')
%%
plot(energy)
title('Energy function for Homework 3.2')
xlabel('Epochs')
ylabel('Energy function')
%%
function [hidden_1_Thresholds, hidden_1_Weights, hidden_2_Weights, hidden_2_Thresholds, hidden_3_Weights, hidden_3_Thresholds,hidden_4_Weights, hidden_4_Thresholds,outputThresholds, outputWeights, u5, u4, u3, u2, u1, energy] = train(tTrain, xTrain, xValid, tValid, xTest, tTest, learning_rate, inputsize, outputsize, batchsize, numberOfEpochs, hiddenNeurons)
       
    %initialization of weights and thresholds
    numberOfWeightsFeedingOutput = hiddenNeurons;
    numberOfWeightsFeedingHidden1 = inputsize;
    numberOfWeightsFeedingHidden = hiddenNeurons;
    outputWeights = initializeWeights(outputsize,hiddenNeurons,numberOfWeightsFeedingOutput);
    hidden_4_Weights = initializeWeights(hiddenNeurons,hiddenNeurons,numberOfWeightsFeedingHidden);
    hidden_3_Weights = initializeWeights(hiddenNeurons,hiddenNeurons,numberOfWeightsFeedingHidden);
    hidden_2_Weights = initializeWeights(hiddenNeurons,hiddenNeurons,numberOfWeightsFeedingHidden);
    hidden_1_Weights = initializeWeights(hiddenNeurons,inputsize,numberOfWeightsFeedingHidden1);
    outputThresholds = zeros(outputsize,1);
    hidden_4_Thresholds = zeros(hiddenNeurons,1);
    hidden_3_Thresholds = zeros(hiddenNeurons,1);
    hidden_2_Thresholds = zeros(hiddenNeurons,1);
    hidden_1_Thresholds = zeros(hiddenNeurons,1);
    
    %shift of the sets
    [xTrain, meanToShift] = shiftTrain(xTrain);
    energy = zeros(numberOfEpochs,1);

    %training start
    for t=1:numberOfEpochs
        t
            totalOutputError = zeros(outputsize,1);
            totalV4Error = zeros(hiddenNeurons,1);
            totalV3Error = zeros(hiddenNeurons,1);
            totalV2Error = zeros(hiddenNeurons,1);
            totalV1Error = zeros(hiddenNeurons,1);
            
            for u=1:size(xTrain,2)
                %feed forward
                X = xTrain(:,u);
                local_field_V1 = hidden_1_Weights*X - hidden_1_Thresholds;
                V1 = sigmoid(local_field_V1);
                local_field_V2 = hidden_2_Weights*V1 - hidden_2_Thresholds;
                V2 = sigmoid(local_field_V2);
                local_field_V3 = hidden_3_Weights*V2 - hidden_3_Thresholds;
                V3 = sigmoid(local_field_V3);
                local_field_V4 = hidden_4_Weights*V3 - hidden_4_Thresholds;
                V4 = sigmoid(local_field_V4);
                local_field_output = outputWeights*V4 - outputThresholds;
                output=sigmoid(local_field_output);
                
                
                energy(t) = energy(t) + sum((tTrain(:,u)-output).^2);

                outputError = (tTrain(:,u)-output) .* sigmoid(local_field_output) .* (1-sigmoid(local_field_output));
                hidden_4_Error = outputWeights' * outputError .* sigmoid(local_field_V4) .* (1-sigmoid(local_field_V4));
                hidden_3_Error = hidden_4_Weights' * hidden_4_Error .* sigmoid(local_field_V3) .* (1-sigmoid(local_field_V3));
                hidden_2_Error = hidden_3_Weights' * hidden_3_Error .* sigmoid(local_field_V2) .* (1-sigmoid(local_field_V2));
                hidden_1_Error = hidden_2_Weights' * hidden_2_Error .* sigmoid(local_field_V1) .* (1-sigmoid(local_field_V1));
                
                totalOutputError = totalOutputError + outputError;
                totalV4Error = totalV4Error + hidden_4_Error;
                totalV3Error = totalV3Error + hidden_3_Error;
                totalV2Error = totalV2Error + hidden_2_Error;
                totalV1Error = totalV1Error + hidden_1_Error;
                
            end
            energy(t) = 0.5*energy(t);
            u5(t) = norm(totalOutputError);
            u4(t) = norm(totalV4Error);
            u3(t) = norm(totalV3Error);
            u2(t) = norm(totalV2Error);
            u1(t) = norm(totalV1Error);

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
            
            delta_hidden_3_Weights = zeros(size(hidden_3_Weights));
            delta_hidden_3_Thresholds = zeros(size(hidden_3_Thresholds));
            
            delta_hidden_4_Weights = zeros(size(hidden_4_Weights));
            delta_hidden_4_Thresholds = zeros(size(hidden_4_Thresholds));
        
            %computation of the output of the batch
            for u=1:size(xBatch,2)
                %feed forward
                X = xBatch(:,u);
                local_field_V1 = hidden_1_Weights*X - hidden_1_Thresholds;
                V1 = sigmoid(local_field_V1);
                local_field_V2 = hidden_2_Weights*V1 - hidden_2_Thresholds;
                V2 = sigmoid(local_field_V2);
                local_field_V3 = hidden_3_Weights*V2 - hidden_3_Thresholds;
                V3 = sigmoid(local_field_V3);
                local_field_V4 = hidden_4_Weights*V3 - hidden_4_Thresholds;
                V4 = sigmoid(local_field_V4);
                local_field_output = outputWeights*V4 - outputThresholds;
                output=sigmoid(local_field_output);

                outputError = (tBatch(:,u)-output) .* sigmoid(local_field_output) .* (1-sigmoid(local_field_output));
                hidden_4_Error = outputWeights' * outputError .* sigmoid(local_field_V4) .* (1-sigmoid(local_field_V4));
                hidden_3_Error = hidden_4_Weights' * hidden_4_Error .* sigmoid(local_field_V3) .* (1-sigmoid(local_field_V3));
                hidden_2_Error = hidden_3_Weights' * hidden_3_Error .* sigmoid(local_field_V2) .* (1-sigmoid(local_field_V2));
                hidden_1_Error = hidden_2_Weights' * hidden_2_Error .* sigmoid(local_field_V1) .* (1-sigmoid(local_field_V1));

                delta_outputWeights = delta_outputWeights + learning_rate * outputError * V4';
                delta_hidden_4_Weights = delta_hidden_4_Weights + learning_rate * hidden_4_Error * V3';
                delta_hidden_3_Weights = delta_hidden_3_Weights + learning_rate * hidden_3_Error * V2';
                delta_hidden_2_Weights = delta_hidden_2_Weights + learning_rate * hidden_2_Error * V1';
                delta_hidden_1_Weights = delta_hidden_1_Weights + learning_rate * hidden_1_Error * X';

                delta_outputThresholds = delta_outputThresholds - learning_rate * outputError;
                delta_hidden_4_Thresholds = delta_hidden_4_Thresholds - learning_rate * hidden_4_Error;
                delta_hidden_3_Thresholds = delta_hidden_3_Thresholds - learning_rate * hidden_3_Error;
                delta_hidden_2_Thresholds = delta_hidden_2_Thresholds - learning_rate * hidden_2_Error;
                delta_hidden_1_Thresholds = delta_hidden_1_Thresholds - learning_rate * hidden_1_Error;

            end

            hidden_1_Weights = hidden_1_Weights + delta_hidden_1_Weights;
            hidden_1_Thresholds = hidden_1_Thresholds + delta_hidden_1_Thresholds;

            hidden_2_Weights = hidden_2_Weights + delta_hidden_2_Weights;
            hidden_2_Thresholds = hidden_2_Thresholds + delta_hidden_2_Thresholds;

            hidden_3_Weights = hidden_3_Weights + delta_hidden_3_Weights;
            hidden_3_Thresholds = hidden_3_Thresholds + delta_hidden_3_Thresholds;

            hidden_4_Weights = hidden_4_Weights + delta_hidden_4_Weights;
            hidden_4_Thresholds = hidden_4_Thresholds + delta_hidden_4_Thresholds;

            outputWeights = outputWeights + delta_outputWeights;
            outputThresholds = outputThresholds + delta_outputThresholds;
        
        p=p+1;
        end
        
    end
          
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

 function processedOutput = postProcess(output)
    [~, winning] = max(output);
    processedOutput = zeros(size(output,1),1);
    processedOutput(winning)=1;
 end