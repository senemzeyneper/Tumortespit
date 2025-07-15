clc;
clear;
close all;

% Dataset ve ön ayarlar
datasetPath = 'C:\Users\senem\Desktop\proje2\brain_tumor_dataset';
imageSize = [224 224 3];

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @customReadFcn);

[imdsTrain, imdsRest] = splitEachLabel(imds, 0.7, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRest, 0.5, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));

% Kombinasyonlar
optimizers = {'adam','adam','sgdm','sgdm','rmsprop','rmsprop'};
lrs = [0.0001, 0.02, 0.0001, 0.02, 0.0001, 0.02];

% MyCNN katmanları
layers = [
    imageInputLayer(imageSize)

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Eğitim döngüsü
for i = 1:length(optimizers)
    opts = trainingOptions(optimizers{i}, ...
        'InitialLearnRate', lrs(i), ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 16, ...
        'ValidationData', imdsValidation, ...
        'ValidationFrequency', 30, ...
        'Plots','training-progress', ...
        'Verbose', false);

    fprintf("MyCNN Model %d eğitiliyor (%s, LR=%.4f)...\n", i, optimizers{i}, lrs(i));
    net = trainNetwork(imdsTrain, layers, opts);

    preds = classify(net, imdsTest);
    acc = mean(preds == imdsTest.Labels);
    figure, confusionchart(imdsTest.Labels, preds);
    title(sprintf('MyCNN - Model %d', i));

    save(sprintf('MyCNN_model_%s_lr%.4f.mat', optimizers{i}, lrs(i)), 'net', 'acc');
end

