clear;
close all;

UsePCA_MDAFeatureReduction = 1 % 0=none, 1=PCA, 2=MDA
% UseClassificationMethod : 0=2D, 1=3D, 2 = ANN2D, 3 = ANN3D, 
%                           4 = Bayesian decision theory, 
%                           5 = GMM2D, 6 = GMM3D, 
%                           7 = GMM2DComp, 8 = GMM3DComp 
UseClassificationMethodStart = 8
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
            case 3 % ANN 3D - select features
                subSet = [1 2 3 4]; % Good
                %subSet = [1 2 3 4 5]; Too many
            case 5 % GMM 2D
                subSet = [1 2];
            case 6 % GMM 3D
                subSet = [1 2 3];                
            case 8 % GMM 3D Comp
                subSet = [1 2 3 4];
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
            case 3 % ANN 3D - select features
                subSet = [1 2 3];
            case 6 % GMM 3D
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
            % 3D classification using Artificial Neural Networks
            [Ctrain, Ctest] = ANN3D(V1new, V1tnew, V2new, V2tnew, Snew, Stnew, size(subSet,2)); % 3 or more features
            ANN3D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            ANN3D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 4
            % Classification based on Bayesian decision theory
            % Assuming a normal distribution of class features
            [t_est, Ctest] = gausianDiscriminant(V1new, V1tnew, V2new, V2tnew); % 2 features only
            %gausianDiscriminant_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            gausianDiscriminant_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 5
            % 2D classification using the Expectation-Maximation (EM)
            % algorithm for Gaussian Mixture Models in 2 dimensions
            % A training is performed for each class V1, V2 and silence
            % finding a Gaussian mixture for each model
            [Ctrain, Ctest] = GMM2D(V1new, V1tnew, V2new, V2tnew, Snew, Stnew); 
            %GMM2D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            GMM2D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 6
            % 3D classification using the Expectation-Maximation (EM)
            % algorithm for Gaussian Mixture Models in 3 dimensions or more
            % A training is performed for each class V1, V2 and silence
            % finding a Gaussian mixture for each model
            [Ctrain, Ctest] = GMM3D(V1new, V1tnew, V2new, V2tnew, Snew, Stnew, size(subSet,2)); % 3 or more features
            %GMM2D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            GMM3D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 7
            % 2D classification using the Expectation-Maximation (EM)
            % algorithm for Gaussian Mixture Models in 2 dimensions
            % A training is performed for each class V1, V2
            % finding Gaussian mixture components for each class
            [Ctrain, Ctest] = GMM2DComponents(V1new, V1tnew, V2new, V2tnew, 4); 
            GMM2DComp_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 8
            % 3D classification using the Expectation-Maximation (EM)
            % algorithm for Gaussian Mixture Models in 2 dimensions
            % A training is performed for each class V1, V2
            % finding Gaussian mixture components for each class
            [Ctrain, Ctest] = GMM3DComponents(V1new, V1tnew, V2new, V2tnew, 4, size(subSet,2)); 
            GMM3DComp_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        otherwise
            % Invalid classification parameter specifier
    end
    
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
