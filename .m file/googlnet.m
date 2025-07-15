% GoogLeNet eğitim scripti (hata düzeltildi)
clc;
clear;
close all;

datasetPath = 'C:\Users\senem\Desktop\proje2\brain_tumor_dataset';
imageSize = [224 224 3];

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @customReadFcn);

[imdsTrain, imdsRest] = splitEachLabel(imds, 0.7, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRest, 0.5, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));

optimizers = {'adam','adam','sgdm','sgdm','rmsprop','rmsprop'};
lrs = [0.0001, 0.02, 0.0001, 0.02, 0.0001, 0.02];

net = googlenet;
lgraph = layerGraph(net);

% Katmanları değiştirme (DÜZELTİLMİŞ)
newFC = fullyConnectedLayer(numClasses,'Name','new_fc');
newSoftmax = softmaxLayer('Name','new_softmax');
newClassOutput = classificationLayer('Name','new_output');

lgraph = replaceLayer(lgraph,'loss3-classifier', newFC);
lgraph = replaceLayer(lgraph,'prob', newSoftmax);
lgraph = replaceLayer(lgraph,'output', newClassOutput);

% Eğitim döngüsü
for i = 1:length(optimizers)
    opts = trainingOptions(optimizers{i}, ...
        'InitialLearnRate', lrs(i), ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 32, ...
        'ValidationData', imdsValidation, ...
        'ValidationFrequency', 30, ...
        'Plots','training-progress', ...
        'Verbose', false);

    fprintf("GoogLeNet Model %d eğitiliyor (%s, LR=%.4f)...\n", i, optimizers{i}, lrs(i));
    netTransfer = trainNetwork(imdsTrain, lgraph, opts);

    preds = classify(netTransfer, imdsTest);
    acc = mean(preds == imdsTest.Labels);
    figure, confusionchart(imdsTest.Labels, preds);
    title(sprintf('GoogLeNet - Model %d', i));

    save(sprintf('GoogLeNet_model_%s_lr%.4f.mat', optimizers{i}, lrs(i)), 'netTransfer', 'acc');
end
