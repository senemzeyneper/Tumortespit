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

net = mobilenetv2;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'Logits','Logits_softmax','ClassificationLayer_Logits'});

newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc')
    softmaxLayer('Name','new_softmax')
    classificationLayer('Name','new_classoutput')];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'global_average_pooling2d_1', 'new_fc');

for i = 1:length(optimizers)
    opts = trainingOptions(optimizers{i}, ...
        'InitialLearnRate', lrs(i), ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 32, ...
        'ValidationData', imdsValidation, ...
        'ValidationFrequency', 30, ...
        'Plots','training-progress', ...
        'Verbose', false);

    fprintf("MobileNetV2 Model %d eğitiliyor (%s, LR=%.4f)...\n", i, optimizers{i}, lrs(i));
    netTransfer = trainNetwork(imdsTrain, lgraph, opts);

    preds = classify(netTransfer, imdsTest);
    acc = mean(preds == imdsTest.Labels);
    figure, confusionchart(imdsTest.Labels, preds);
    title(sprintf('MobileNetV2 - Model %d', i));

    save(sprintf('MobileNetV2_model_%s_lr%.4f.mat', optimizers{i}, lrs(i)), 'netTransfer', 'acc');
end
