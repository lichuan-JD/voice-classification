clear;

UsePCA_MDAFeatureReduction = 1 % 0=none, 1=PCA, 2=MDA
% UseClassificationMethod : 0=2D, 1=3D, 2 = ANN2D, 
%                           3 = Bayesian decision theory, 
%                           4 = GMM2D, 5 = GMM3D, 
%                           6 = GMM2DComp, 
%                           7 = ANN3D, 8 = GMM3DComp 
UseClassificationMethodStart = 7
UseClassificationMethodEnd = 8
%UseSizeTrainSet = 94 % Op or Ned or same speech
%UseSizeTestSet = 94

%UseSizeTrainSet = 188 % Op and Ned or same speech
%UseSizeTestSet = 188
%UseSizeTrainSet = 377 % Same speech twice
%UseSizeTestSet = 377
UseSizeTrainSet = 1700 % All samples for GMM2DComp
UseSizeTestSet = 150
UseRandomisation = 2 % 0, 1, 2 (Mixed)
EndLoopDimensions = 12 % 3-12

% Clear results
GMMtest_error(2) = 0.36;
ANNtest_error(2) = 0.35;

% Loop dimensions
for dimensions = 3:EndLoopDimensions
    
close all;
    
% Start, End
% 0,1  Op/Ned
% 2,2  Same speech
% 2,3  Same speech twice
% 2,5  All speech
% 0,5  All
%[mfcc_voice1 mfcc_voice2 mfcc_silence] = CreateMFCCSamples(0, 0, 0, 0); % Op
%[mfcc_voice1 mfcc_voice2 mfcc_silence] = CreateMFCCSamples(1, 0, 1, 1); % Ned
%[mfcc_voice1 mfcc_voice2 mfcc_silence] = CreateMFCCSamples(1, 0, 2, 2);
%[mfcc_voice1 mfcc_voice2 mfcc_silence] = CreateMFCCSamples(1, 0, 0, 5);
%[mfcc_voice1 mfcc_voice2 mfcc_silence] = CreateMFCCSamples(1, 0, 2, 5);
[mfcc_voice1 mfcc_voice2 mfcc_silence] = CreateMFCCSamples(0, 0, 0, 5);

features = size(mfcc_voice1, 2)
samples_total = size(mfcc_voice1, 1)

data_rand = randperm(samples_total); %Randomize feature set training data
if UseRandomisation == 1
    train_rand = data_rand(1:UseSizeTrainSet);
    test_rand = data_rand(UseSizeTrainSet+1:(UseSizeTestSet+UseSizeTrainSet));
else
    train_rand = 1:UseSizeTrainSet;
    test_rand = UseSizeTrainSet+1:(UseSizeTestSet+UseSizeTrainSet);
end
if UseRandomisation == 2
   test_rand = 377+1:2*377;
end

mfcc_v1 = mfcc_voice1(train_rand, :);
mfcc_s = mfcc_silence(train_rand, :); 
mfcc_v2 = mfcc_voice2(train_rand, :); 

mfcc_v1t = mfcc_voice1(test_rand, :);
mfcc_st = mfcc_silence(test_rand, :); 
mfcc_v2t = mfcc_voice2(test_rand, :); 

%% Loop all methods
firstLoop = 1;
for UseClassificationMethod = UseClassificationMethodStart:UseClassificationMethodEnd

    %% PCA or MDA feature reduction
    if UsePCA_MDAFeatureReduction == 1
        % PCA feature reduction
        switch UseClassificationMethod
            case 1 % Linear 3D
                subSet = [1 2 3];
            case 4 % GMM 2D
                subSet = [1 2];
            case 5 % GMM 3D
                subSet = [1 2 3];                
            case 7 % ANN 3D - select features
                subSet = 1:dimensions;
            case 8 % GMM 3D Comp
                subSet = 1:dimensions; 
            otherwise
                subSet = [1 2];
        end
        [v1] = PrincipalComponentAnalysis(mfcc_v1, subSet); % Sub set of principal components
    else
    if UsePCA_MDAFeatureReduction == 2
        % MDA feature reduction
        switch UseClassificationMethod
            case 1 % Linear 3D
                subSet = [1 2 3];
            case 5 % GMM 3D
                subSet = [1 2 3];
            case 7 % ANN 3D - select features
                subSet = [1 2 3];
            case 8 % GMM 3D Comp - select features
                subSet = [1 2 3];
            otherwise
                subSet = [1 2];
        end
        [v1] = MultipleDiscriminantAnalysis(mfcc_v1, mfcc_s, mfcc_v2, subSet);
    else
        % None feature reduction
        subSet = 1:features;
        v1 = 1;
    end
    end
    
    V1new = mfcc_v1*v1; % projecting onto the new basis
    V2new = mfcc_v2*v1; % projection on the same basis..
    Snew = mfcc_s*v1; % projection on the same basis..
    
    V1tnew = mfcc_v1t*v1; % projecting on the same basis
    V2tnew = mfcc_v2t*v1; % projection on the same basis..
    Stnew = mfcc_st*v1; % projection on the same basis..
    
    %% Plot projected features
    if firstLoop == 1 
        if size(subSet,2) > 2 
            figure;
            scatter3(V1new(:,1), V1new(:,2), V1new(:,3), 'r.');
            hold on;
            scatter3(Snew(:,1), Snew(:,2), Snew(:,3), 'b.');
            scatter3(V2new(:,1), V2new(:,2), V2new(:,3), 'g.');
            title('Projection training data (Voice1 - red, Silence - blue, Voice2 - green)');
            xlabel('e1');
            ylabel('e2');
            zlabel('e3');
            figure;
            scatter3(V1tnew(:,1), V1tnew(:,2), V1tnew(:,3), 'r.');
            hold on;
            scatter3(V2tnew(:,1), V2tnew(:,2), V2tnew(:,3), 'g.');
            title('Projection test data (Voice1 - red, Voice2 - green)');
            xlabel('e1');
            ylabel('e2');
            zlabel('e3');
        else % 2 dimensions
            figure;
            scatter(V1new(:,1), V1new(:,2), 'r.');
            hold on;
            scatter(Snew(:,1), Snew(:,2), 'b.');
            scatter(V2new(:,1), V2new(:,2), 'g.');
            title('Projection training data (Voice1 - red, Silence - blue, Voice2 - green)');
            xlabel('e1');
            ylabel('e2');
            figure;
            scatter(V1tnew(:,1), V1tnew(:,2), 'r.');
            hold on;
            scatter(V2tnew(:,1), V2tnew(:,2), 'g.');
            title('Projection test data (Voice1 - red, Voice2 - green)');
            xlabel('e1');
            ylabel('e2');
        end
        firstLoop = 0;
    end
    
    %% Classification of test set with 2 classes and 2 or 3 features
    
    switch (UseClassificationMethod)
        case 0
            % 2D classification training set with 2 classes and 2 features
            [Ctrain, Ctest, W] = linear2D(V1new, V1tnew, V2new, V2tnew); % training
            linear2D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            linear2D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 1
            % 3D classification training set with 2 classes and 3 features
            % or more
            [Ctrain, Ctest, W] = linear3D(V1new, V1tnew, V2new, V2tnew); % training
            linear3D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            linear3D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 2
            % 2D classification using Artificial Neural Networks
            [Ctrain, Ctest] = ANN2D(V1new, V1tnew, V2new, V2tnew, Snew, Stnew, 2); % 2 or 3 features
            ANN2D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            ANN2D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 3
            % Classification based on Bayesian decision theory
            % Assuming a normal distribution of class features
            [t_est, Ctest] = gausianDiscriminant(V1new, V1tnew, V2new, V2tnew); % 2 features only
            %gausianDiscriminant_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            gausianDiscriminant_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 4
            % 2D classification using the Expectation-Maximation (EM)
            % algorithm for Gaussian Mixture Models in 2 dimensions
            % A training is performed for each class V1, V2 and silence
            % finding a Gaussian mixture for each model
            [Ctrain, Ctest] = GMM2D(V1new, V1tnew, V2new, V2tnew, Snew, Stnew); 
            %GMM2D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            GMM2D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 5
            % 3D classification using the Expectation-Maximation (EM)
            % algorithm for Gaussian Mixture Models in 3 dimensions or more
            % A training is performed for each class V1, V2 and silence
            % finding a Gaussian mixture for each model
            [Ctrain, Ctest] = GMM3D(V1new, V1tnew, V2new, V2tnew, Snew, Stnew, size(subSet,2)); % 3 or more features
            %GMM2D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            GMM3D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 6
            % 2D classification using the Expectation-Maximation (EM)
            % algorithm for Gaussian Mixture Models in 2 dimensions
            % A training is performed for each class V1, V2
            % finding Gaussian mixture components for each class
            [Ctrain, Ctest] = GMM2DComponents(V1new, V1tnew, V2new, V2tnew, 4); 
            GMM2DComp_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 7
            % 3D classification using Artificial Neural Networks
            [Ctrain, Ctest] = ANN3D(V1new, V1tnew, V2new, V2tnew, Snew, Stnew, size(subSet,2)); % 3 or more features
            ANN3D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            ANN3D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
            ANNtest_error(dimensions) = ANN3D_test;
        case 8
            % 3D classification using the Expectation-Maximation (EM)
            % algorithm for Gaussian Mixture Models in 2 dimensions
            % A training is performed for each class V1, V2
            % finding Gaussian mixture components for each class
            [Ctrain, Ctest] = GMM3DComponents(V1new, V1tnew, V2new, V2tnew, 4, size(subSet,2)); 
            GMM3DComp_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
            GMMtest_error(dimensions) = GMM3DComp_test;
        otherwise
            % Invalid classification parameter specifier
    end
    
end

end

if UseClassificationMethodStart == 7 && UseClassificationMethodEnd == 8
    figure, hold on,
    plot(GMMtest_error, '*r');
    plot(ANNtest_error, 'ob');
    hold off;
    title('ANN(o) and GMM(*) classifcation error vs. dimensions');
    xlabel('dimensions');
    ylabel('test error');
    save GMMtest_error;
    save ANNtest_error;
end

%% Printing final results
if UseClassificationMethodStart == 0 && UseClassificationMethodEnd == 8
    ANN2D_train
    ANN2D_test
    
    linear2D_train
    linear2D_test
    
    linear3D_train
    linear3D_test
    
    ANN3D_train
    ANN3D_test
    
    gausianDiscriminant_test
    GMM2D_test
    GMM3D_test
    
    GMM2DComp_test
    GMM3DComp_test
end
