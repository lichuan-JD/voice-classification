clear;
close all;

UsePCA_MDAFeatureReduction = 1 % 0=PCA, 1=MDA
UseClassificationMethodEnd = 4
UseSizeTrainSet = 100
UseSizeTestSet = UseSizeTrainSet

[mfcc_voice1 mfcc_voice2 mfcc_silence] = CreateMFCCSamples();

features = size(mfcc_voice1, 2)
samples_total = size(mfcc_voice1, 1)

train_rand = randperm(UseSizeTrainSet); %Randomize feature set training data
mfcc_v1 = mfcc_voice1(train_rand, :); 
mfcc_s = mfcc_silence(train_rand, :); 
mfcc_v2 = mfcc_voice2(train_rand, :); 

test_rand = randperm(UseSizeTestSet); %Randomize feature set test data
mfcc_v1t = mfcc_voice1(test_rand, :);
mfcc_v2t = mfcc_voice2(test_rand, :); 

%% Loop all methods
% UseClassificationMethod : 0=2D, 1=3D, 2 = ANN2D, 3 = ANN3D, 4 = Bayesian decision theory
for UseClassificationMethod = 0:UseClassificationMethodEnd

    %% PCA or MDA feature reduction
    if UsePCA_MDAFeatureReduction == 0
        % PCA feature reduction
        subSet = [1 2 3];
        [v1] = PrincipalComponentAnalysis(mfcc_v1, subSet); % Sub set of principal components
    else
        % MDA feature reduction
        switch UseClassificationMethod
            case 0 % Linear 2D
                subSet = [1 2];
            case 1 % Linear 3D
                subSet = [1 2 3];
            case 3 % ANN 3D - select features
                subSet = [1 2 3]; % Good
                %subSet = [1 2 3 4 5]; Too many
            case 4 % Baysian decision theory
                subSet = [1 2];
            otherwise
                subSet = [1 2 3];
        end
        [v1] = MultipleDiscriminantAnalysis(mfcc_v1, mfcc_s, mfcc_v2, subSet);
    end
    
    V1new = mfcc_v1*v1; % projecting onto the new basis
    V2new = mfcc_v2*v1; % projection on the same basis..
    Snew = mfcc_s*v1; % projection on the same basis..
    
    V1tnew = mfcc_v1t*v1; % projecting on the same basis
    V2tnew = mfcc_v2t*v1; % projection on the same basis..
    
    %% Plot projected features
    if (size(subSet,2) > 2)
        figure(1);
        scatter3(V1new(:,1), V1new(:,2), V1new(:,3), 'r.');
        hold on;
        scatter3(Snew(:,1), Snew(:,2), Snew(:,3), 'b.');
        scatter3(V2new(:,1), V2new(:,2), V2new(:,3), 'g.');
        title('Projection of MFCC (Voice1 - red, Silence - blue, Voice2 - green)');
        xlabel('e1');
        ylabel('e2');
        zlabel('e3');
    end
    
    %% Classification of test set with 2 classes and 2 or 3 features
    
    switch (UseClassificationMethod)
        case 0
            % 2D classification training set with 2 classes and 2 features
            [Ctrain, Ctest, W] = linear2D(V1new, V2new, V1tnew, V2tnew); % training
            linear2D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            linear2D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 1
            % 3D classification training set with 2 classes and 3 features
            [Ctrain, Ctest, W] = linear3D(V1new, V2new, V1tnew, V2tnew); % training
            linear3D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            linear3D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 2
            % 2D classification using Artificial Neural Networks
            [Ctrain, Ctest] = ANN2D(V1new, V1tnew, V2new, V2tnew, Snew, 3); % 2 or 3 features
            ANN2D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            ANN2D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 3
            % 3D classification using Artificial Neural Networks
            [Ctrain, Ctest] = ANN3D(V1new, V1tnew, V2new, V2tnew, Snew, size(subSet,2));
            ANN3D_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            ANN3D_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        case 4
            % Classification based on bayesian decision theory
            % Assuming a normal distribution of class features
            [t_est, Ctest] = gausianDiscriminant(V1new, V1tnew, V2new, V2tnew); % 2 features only
            %gausianDiscriminant_train = 1-sum(diag(Ctrain))/sum(Ctrain(:)); % correct classification percentage
            gausianDiscriminant_test= 1-sum(diag(Ctest))/sum(Ctest(:)); % correct classification percentage
        otherwise
            % Invalid classification parameter specifier
    end
    
end

%% Printing final results
linear2D_train
linear2D_test

linear3D_train
linear3D_test

ANN2D_train
ANN2D_test

ANN3D_train
ANN3D_test

%gausianDiscriminant_train
gausianDiscriminant_test
