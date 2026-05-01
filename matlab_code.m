clc;
clear;
close all;

%% ==============================
%  CVD-CDSS ALL-IN-ONE MATLAB CODE
% ===============================

disp('1. Fetching data from UCI Repository...');

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data';

varNames = {'Age','Sex','ChestPain','RestingBP','Cholesterol', ...
    'FastingBloodSugar','RestECG','MaxHeartRate','ExerciseAngina', ...
    'ST_Depression','ST_Slope','MajorVessels','Thalassemia','Target'};

opts = detectImportOptions(url,'FileType','text','ReadVariableNames',false);
opts.VariableNames = varNames;

data = readtable(url, opts);

%% ==============================
%  DATA CLEANING
% ==============================
disp('2. Cleaning data...');

if iscell(data.MajorVessels)
    data.MajorVessels = str2double(data.MajorVessels);
end

if iscell(data.Thalassemia)
    data.Thalassemia = str2double(data.Thalassemia);
end

data = rmmissing(data);
data.Target = double(data.Target > 0);

%% ==============================
%  DATA SPLIT
% ==============================
disp('3. Splitting data (80/20)...');

X = data(:,1:end-1);
Y = data.Target;

cv = cvpartition(size(data,1),'HoldOut',0.2);

XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);

XTest = X(test(cv), :);
YTest = Y(test(cv), :);

%% ==============================
%  MODEL 1: LOGISTIC REGRESSION
% ==============================
disp('4. Training Logistic Regression...');
mdl_LR = fitglm(XTrain, YTrain, 'Distribution','binomial','Link','logit');

pred_LR = double(predict(mdl_LR, XTest) > 0.5);
acc_LR = sum(pred_LR == YTest) / length(YTest);

%% ==============================
%  MODEL 2: SVM
% ==============================
disp('5. Training SVM...');
mdl_SVM = fitcsvm(XTrain, YTrain, 'KernelFunction','rbf','Standardize',true);

pred_SVM = predict(mdl_SVM, XTest);
acc_SVM = sum(pred_SVM == YTest) / length(YTest);

%% ==============================
%  MODEL 3: RANDOM FOREST
% ==============================
disp('6. Training Random Forest...');
mdl_RF = fitcensemble(XTrain, YTrain, 'Method','Bag','NumLearningCycles',100);

pred_RF = predict(mdl_RF, XTest);
acc_RF = sum(pred_RF == YTest) / length(YTest);

%% ==============================
%  RESULTS
% ==============================
fprintf('\n====================================\n');
fprintf('MODEL RESULTS:\n');
fprintf('Logistic Regression: %.2f%%\n', acc_LR*100);
fprintf('SVM: %.2f%%\n', acc_SVM*100);
fprintf('Random Forest: %.2f%%\n', acc_RF*100);
fprintf('====================================\n');

%% ==============================
%  SELECT BEST MODEL
% ==============================
models = {mdl_LR, mdl_SVM, mdl_RF};
accuracies = [acc_LR, acc_SVM, acc_RF];

[bestAcc, bestIdx] = max(accuracies);
bestModel = models{bestIdx};

fprintf('\nBest Model Selected: Model %d with %.2f%% accuracy\n', bestIdx, bestAcc*100);

%% ==============================
%  VISUALIZATION
% ==============================
figure;
bar(categorical({'Logistic Regression','SVM','Random Forest'}), accuracies);
ylim([0.5 1]);
ylabel('Accuracy');
title('Model Comparison');

%% ==============================
%  LIVE PREDICTION (INPUT SECTION)
% ==============================
disp(' ');
disp('===== ENTER PATIENT DATA =====');

% Example Input (you can change values)
input_array = [52 1 3 140 250 0 1 150 0 1.2 2 0 3];

varNamesInput = {'Age','Sex','ChestPain','RestingBP','Cholesterol', ...
    'FastingBloodSugar','RestECG','MaxHeartRate','ExerciseAngina', ...
    'ST_Depression','ST_Slope','MajorVessels','Thalassemia'};

patientData = array2table(input_array,'VariableNames',varNamesInput);

%% ==============================
%  PREDICTION
% ==============================
probability = predict(bestModel, patientData);

fprintf('\nPredicted Heart Disease Probability: %.2f\n', probability);

%% ==============================
%  TRIAGE OUTPUT
% ==============================
if probability >= 0.85
    disp('RISK LEVEL: CRITICAL 🚨');
elseif probability >= 0.40
    disp('RISK LEVEL: ELEVATED ⚠️');
else
    disp('RISK LEVEL: LOW ✅');
end