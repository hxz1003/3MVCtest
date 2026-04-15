%% GRID_SEARCH_CALTECH256 对 Caltech256 执行联合加权融合网格搜索
% 功能简介：
% 默认对 dataset/Caltech256.mat 数据集搜索 beta、lambda、rho 和 tauS，
% 并以 ACC 最大作为最优参数组合选择标准。考虑到 Caltech256 是新接入数据，
% 本脚本在不修改 .mat 文件本身的前提下，支持在内存中完成标签字段适配、
% 视图方向修正、可选移除 clutter 类别以及可选按类限样。
%
% 输入参数说明：
% 本脚本无函数输入。可直接修改“用户可配置参数”区域中的搜索网格、随机种子
% 和 adjustmentOptions；也可在命令行预先定义同名变量后再运行脚本，以覆盖默认值。
%
% 输出参数说明：
% 运行结束后，工作区将生成：
%   searchRecords - 每一次参数搜索的详细结果结构体数组
%   searchTable   - searchRecords 对应的结果表
%   comboTable    - 参数组合汇总结果表
%   bestResult    - ACC 最优的搜索结果结构体
%   datasetInfo   - 当前数据集及内存级调整信息结构体
%
% 维度说明：
% X 为 v x 1 或 1 x v 的 cell，每个视图为 n x d_v；
% Y 为 n x 1 标签向量。
%
% 注意事项：
% 1. 该脚本默认启用 3.Y Quality-and-Alignment Aware Weighted Anchor Fusion。
% 2. Caltech256 规模较大，默认搜索网格为首轮粗搜索，repeatNum 默认设为 1。
% 3. 默认会在内存中移除疑似 clutter 类别；若需保留全部 257 类，可将
%    adjustmentOptions.removeClutterClass 改为 false。
%
% See also algo_qp, aligned, Neighbor, myNMIACCwithmean

clearvars -except datasetFile betaList lambdaList rhoList tauSList epsilonValue ...
    repeatNum baseSeed labelFieldCandidates adjustmentOptions
clc;
warning off;

%% 用户可配置参数
if ~exist('datasetFile', 'var') || isempty(datasetFile)
    datasetFile = 'Caltech256.mat';
end
if ~exist('betaList', 'var') || isempty(betaList)
    % Caltech256 规模较大，默认先围绕 demo.m 的 beta=100 做粗搜索。
    betaList = [80, 100, 120];
end
if ~exist('lambdaList', 'var') || isempty(lambdaList)
    lambdaList = [5e3, 1e4, 2e4];
end
if ~exist('rhoList', 'var') || isempty(rhoList)
    rhoList = [0.3, 0.5, 0.7];
end
if ~exist('tauSList', 'var') || isempty(tauSList)
    tauSList = [0.5, 1];
end
if ~exist('epsilonValue', 'var') || isempty(epsilonValue)
    epsilonValue = 1e-8;
end
if ~exist('repeatNum', 'var') || isempty(repeatNum)
    repeatNum = 1;
end
if ~exist('baseSeed', 'var') || isempty(baseSeed)
    baseSeed = 1;
end
if ~exist('labelFieldCandidates', 'var') || isempty(labelFieldCandidates)
    labelFieldCandidates = {'Y', 'y', 'gnd', 'gt', 'truth', 'label', 'labels'};
end
if ~exist('adjustmentOptions', 'var') || isempty(adjustmentOptions)
    adjustmentOptions = struct();
end

%% 环境初始化
projectRoot = fileparts(mfilename('fullpath'));
datasetDir = fullfile(projectRoot, 'dataset');
addpath(genpath(projectRoot));

adjustmentOptions = normalize_adjustment_options(adjustmentOptions);
validate_search_config(datasetFile, betaList, lambdaList, rhoList, tauSList, ...
    epsilonValue, repeatNum, baseSeed, labelFieldCandidates, adjustmentOptions);
[X, Y, datasetInfo] = load_multiview_dataset_with_adjustment(datasetDir, datasetFile, ...
    labelFieldCandidates, adjustmentOptions);
k = datasetInfo.classNum;
viewNum = datasetInfo.viewNum;
metricNames = {'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy'};
totalSearchNum = numel(betaList) * numel(lambdaList) * numel(rhoList) * numel(tauSList) * repeatNum;

fprintf('开始网格搜索：数据集=%s | 数据集别名=%s | 标签字段=%s\n', ...
    datasetInfo.datasetFile, datasetInfo.datasetName, datasetInfo.labelField);
fprintf('原始样本数=%d | 原始类别数=%d | 调整后样本数=%d | 调整后视图数=%d | 调整后类别数=%d\n', ...
    datasetInfo.originalSampleNum, datasetInfo.originalClassNum, datasetInfo.sampleNum, ...
    datasetInfo.viewNum, datasetInfo.classNum);
fprintf('内存级数据调整：%s\n', datasetInfo.adjustmentSummary);
fprintf('搜索网格：beta=%s | lambda=%s | rho=%s | tauS=%s | repeatNum=%d | baseSeed=%d\n', ...
    mat2str(betaList), mat2str(lambdaList), mat2str(rhoList), ...
    mat2str(tauSList), repeatNum, baseSeed);

searchRecords(totalSearchNum, 1) = create_empty_record(metricNames, viewNum);
bestResult = create_empty_record(metricNames, viewNum);
bestResult.ACC = -inf;

comboSummary = repmat(create_empty_combo_summary(viewNum), ...
    numel(betaList) * numel(lambdaList) * numel(rhoList) * numel(tauSList), 1);
searchIdx = 0;
comboIdx = 0;

for ib = 1:numel(betaList)
    beta = betaList(ib);
    for il = 1:numel(lambdaList)
        lambda = lambdaList(il);
        for irho = 1:numel(rhoList)
            rho = rhoList(irho);
            for its = 1:numel(tauSList)
                tauS = tauSList(its);
                comboIdx = comboIdx + 1;
                comboACC = nan(repeatNum, 1);
                comboTime = nan(repeatNum, 1);
                comboWeights = nan(repeatNum, viewNum);
                comboAlignmentErrors = nan(repeatNum, viewNum);

                for ir = 1:repeatNum
                    currentSeed = baseSeed + ir - 1;
                    rng(currentSeed, 'twister');
                    tic;

                    targetView = NaN;
                    iterNum = NaN;
                    obj = [];
                    thetaAll = cell(viewNum, 1);
                    neighborTime = nan(viewNum, 1);
                    qualityScores = nan(viewNum, 1);
                    metricMean = nan(1, numel(metricNames));
                    metricStd = nan(1, numel(metricNames));
                    fusionInfo = create_empty_fusion_info(viewNum, rho, tauS, epsilonValue);
                    status = 'success';
                    errorMessage = '';

                    try
                        for iv = 1:viewNum
                            [~, timeNeighbor, ~, object, theta] = Neighbor(X{iv}, Y);
                            thetaAll{iv, 1} = theta;
                            qualityScores(iv, 1) = sum(object);
                            neighborTime(iv, 1) = timeNeighbor;
                        end

                        [~, targetView] = min(qualityScores);
                        fusionOptions = struct();
                        fusionOptions.mode = 'quality_alignment_weighted';
                        fusionOptions.qualityScores = qualityScores;
                        fusionOptions.rho = rho;
                        fusionOptions.tauS = tauS;
                        fusionOptions.epsilon = epsilonValue;

                        [U, ~, ~, iterNum, obj, fusionInfo] = algo_qp(X, Y, thetaAll, ...
                            beta, lambda, targetView, fusionOptions);
                        [metricMean, metricStd] = myNMIACCwithmean(U, Y, k);

                        if isempty(obj) || any(~isfinite(obj))
                            error('当前参数组合产生了非法目标函数值，请检查 beta、lambda、rho 或 tauS。');
                        end
                        if any(~isfinite(metricMean))
                            error('当前参数组合产生了非法评价指标，请检查算法输出是否稳定。');
                        end
                    catch ME
                        status = 'failed';
                        errorMessage = ME.message;
                    end

                    elapsedTime = toc;
                    searchIdx = searchIdx + 1;
                    searchRecords(searchIdx) = build_search_record(datasetInfo, metricNames, viewNum, ...
                        beta, lambda, rho, tauS, epsilonValue, ir, currentSeed, targetView, ...
                        iterNum, obj, sum(neighborTime(isfinite(neighborTime))), elapsedTime, ...
                        qualityScores, fusionInfo.alignmentErrors, fusionInfo.jointScores, ...
                        fusionInfo.weights, metricMean, metricStd, status, errorMessage);

                    comboACC(ir, 1) = metricMean(1);
                    comboTime(ir, 1) = elapsedTime;
                    comboWeights(ir, :) = fusionInfo.weights(:)';
                    comboAlignmentErrors(ir, :) = fusionInfo.alignmentErrors(:)';

                    if strcmp(status, 'success')
                        fprintf(['搜索 %3d/%3d | beta=%-8g | lambda=%-8g | rho=%.2f | tauS=%-6g | ' ...
                            'repeat=%d/%d | seed=%d | ACC=%.6f | Time=%.2fs\n'], ...
                            searchIdx, totalSearchNum, beta, lambda, rho, tauS, ...
                            ir, repeatNum, currentSeed, metricMean(1), elapsedTime);
                        fprintf('  质量分数: %s\n', vector_to_text(qualityScores));
                        fprintf('  对齐残差: %s\n', vector_to_text(fusionInfo.alignmentErrors));
                        fprintf('  联合得分: %s\n', vector_to_text(fusionInfo.jointScores));
                        fprintf('  视图权重: %s\n', vector_to_text(fusionInfo.weights));

                        if metricMean(1) > bestResult.ACC || ...
                                (abs(metricMean(1) - bestResult.ACC) <= eps && elapsedTime < bestResult.elapsedTime)
                            bestResult = searchRecords(searchIdx);
                        end
                    else
                        fprintf(['搜索 %3d/%3d | beta=%-8g | lambda=%-8g | rho=%.2f | tauS=%-6g | ' ...
                            'repeat=%d/%d | seed=%d | 失败 | Time=%.2fs | 错误=%s\n'], ...
                            searchIdx, totalSearchNum, beta, lambda, rho, tauS, ...
                            ir, repeatNum, currentSeed, elapsedTime, errorMessage);
                    end
                end

                comboSummary(comboIdx) = build_combo_summary(comboSummary(comboIdx), beta, lambda, ...
                    rho, tauS, comboACC, comboTime, comboWeights, comboAlignmentErrors);

                fprintf(['组合汇总 | beta=%-8g | lambda=%-8g | rho=%.2f | tauS=%-6g | ' ...
                    '最佳ACC=%.6f | 平均ACC=%.6f | 平均耗时=%.2fs\n'], ...
                    beta, lambda, rho, tauS, comboSummary(comboIdx).bestACC, ...
                    comboSummary(comboIdx).meanACC, comboSummary(comboIdx).meanTime);
                fprintf('  平均对齐残差: %s\n', comboSummary(comboIdx).meanAlignmentErrorsText);
                fprintf('  平均视图权重: %s\n', comboSummary(comboIdx).meanWeightsText);
            end
        end
    end
end

searchTable = struct2table(searchRecords, 'AsArray', true);
comboTable = struct2table(comboSummary, 'AsArray', true);

if isfinite(bestResult.ACC)
    fprintf(['\n最优参数组合：beta=%g | lambda=%g | rho=%.2f | tauS=%g | ' ...
        'ACC=%.6f | NMI=%.6f | Time=%.2fs | seed=%d\n'], ...
        bestResult.beta, bestResult.lambda, bestResult.rho, bestResult.tauS, ...
        bestResult.ACC, bestResult.NMI, bestResult.elapsedTime, bestResult.seed);
    fprintf('数据调整摘要：%s\n', bestResult.adjustmentSummary);
    fprintf('对应视图质量分数：%s\n', bestResult.qualityScoresText);
    fprintf('对应对齐残差：%s\n', bestResult.alignmentErrorsText);
    fprintf('对应联合得分：%s\n', bestResult.jointScoresText);
    fprintf('对应视图权重：%s\n', bestResult.viewWeightsText);
    fprintf(['其余指标：Purity=%.6f | Fscore=%.6f | Precision=%.6f | ' ...
        'Recall=%.6f | AR=%.6f | Entropy=%.6f\n'], ...
        bestResult.Purity, bestResult.Fscore, bestResult.Precision, ...
        bestResult.Recall, bestResult.AR, bestResult.Entropy);
else
    warning('所有参数组合均搜索失败，请检查数据集、随机种子或搜索网格范围。');
end


function adjustmentOptions = normalize_adjustment_options(adjustmentOptions)
% NORMALIZE_ADJUSTMENT_OPTIONS 规范化 Caltech256 的内存级数据调整配置。

if ~isstruct(adjustmentOptions)
    error('adjustmentOptions 必须是结构体。');
end

adjustmentOptions = ensure_default_field(adjustmentOptions, 'removeClutterClass', true);
adjustmentOptions = ensure_default_field(adjustmentOptions, 'selectedLabels', []);
adjustmentOptions = ensure_default_field(adjustmentOptions, 'maxSamplesPerClass', []);
adjustmentOptions = ensure_default_field(adjustmentOptions, 'adjustmentSeed', 1);
adjustmentOptions = ensure_default_field(adjustmentOptions, 'remapLabels', true);
adjustmentOptions = ensure_default_field(adjustmentOptions, 'autoTransposeViews', true);
end


function structValue = ensure_default_field(structValue, fieldName, defaultValue)
% ENSURE_DEFAULT_FIELD 为缺失字段补齐默认值。

if ~isfield(structValue, fieldName) || isempty(structValue.(fieldName))
    structValue.(fieldName) = defaultValue;
end
end


function fusionInfo = create_empty_fusion_info(viewNum, rho, tauS, epsilonValue)
% CREATE_EMPTY_FUSION_INFO 创建融合信息占位结构体。

fusionInfo = struct();
fusionInfo.mode = 'quality_alignment_weighted';
fusionInfo.targetView = NaN;
fusionInfo.qualityScores = nan(viewNum, 1);
fusionInfo.alignmentErrors = nan(viewNum, 1);
fusionInfo.normalizedQualityScores = nan(viewNum, 1);
fusionInfo.normalizedAlignmentErrors = nan(viewNum, 1);
fusionInfo.jointScores = nan(viewNum, 1);
fusionInfo.weights = nan(viewNum, 1);
fusionInfo.rho = rho;
fusionInfo.tauS = tauS;
fusionInfo.epsilon = epsilonValue;
end


function [X, Y, datasetInfo] = load_multiview_dataset_with_adjustment(datasetDir, datasetFile, ...
    labelFieldCandidates, adjustmentOptions)
% LOAD_MULTIVIEW_DATASET_WITH_ADJUSTMENT 加载并在内存中适配多视图数据。

datasetPath = fullfile(datasetDir, datasetFile);
if exist(datasetPath, 'file') ~= 2
    error('未找到数据集文件：%s。请确认 datasetFile 是否正确。', datasetPath);
end

datasetStruct = load(datasetPath);
if ~isfield(datasetStruct, 'X')
    error('数据集 %s 中缺少变量 X，无法继续运行。', datasetFile);
end

datasetName = resolve_dataset_name(datasetStruct, datasetFile);
X = datasetStruct.X;
if ~iscell(X) || isempty(X)
    error('数据集 %s 中的 X 必须是非空 cell 数组。', datasetFile);
end
X = X(:);

labelField = resolve_label_field(datasetStruct, labelFieldCandidates, datasetFile);
Y = datasetStruct.(labelField);
Y = double(Y(:));

[X, orientationSummary] = maybe_transpose_views(X, Y, datasetFile, adjustmentOptions.autoTransposeViews);
[X, Y] = validate_multiview_data(X, Y, datasetFile);

originalSampleNum = numel(Y);
originalClassNum = numel(unique(Y));

[X, Y, adjustmentSummary] = apply_dataset_adjustment(X, Y, datasetFile, adjustmentOptions, orientationSummary);
[X, Y] = validate_multiview_data(X, Y, datasetFile);

datasetInfo = struct();
datasetInfo.datasetFile = datasetFile;
datasetInfo.datasetName = datasetName;
datasetInfo.datasetPath = datasetPath;
datasetInfo.labelField = labelField;
datasetInfo.originalSampleNum = originalSampleNum;
datasetInfo.originalClassNum = originalClassNum;
datasetInfo.sampleNum = numel(Y);
datasetInfo.viewNum = numel(X);
datasetInfo.classNum = numel(unique(Y));
datasetInfo.adjustmentSummary = adjustmentSummary;
end


function datasetName = resolve_dataset_name(datasetStruct, datasetFile)
% RESOLVE_DATASET_NAME 解析数据集别名，便于日志输出。

datasetName = erase(datasetFile, '.mat');
if isfield(datasetStruct, 'data_name') && ~isempty(datasetStruct.data_name)
    rawName = datasetStruct.data_name;
    if iscell(rawName) && ~isempty(rawName) && (ischar(rawName{1}) || isstring(rawName{1}))
        datasetName = char(string(rawName{1}));
    elseif ischar(rawName) || (isstring(rawName) && isscalar(rawName))
        datasetName = char(string(rawName));
    end
end
end


function [X, summaryText] = maybe_transpose_views(X, Y, datasetFile, autoTransposeViews)
% MAYBE_TRANSPOSE_VIEWS 若视图写成 d x n，则在内存中自动转为 n x d。

if ~autoTransposeViews
    summaryText = '未启用视图自动转置检查';
    return;
end

sampleNum = numel(Y);
summaryParts = cell(numel(X), 1);
summaryCount = 0;
for iv = 1:numel(X)
    Xi = X{iv};
    if ~(isnumeric(Xi) || islogical(Xi))
        error('数据集 %s 的第 %d 个视图必须是数值矩阵。', datasetFile, iv);
    end
    if size(Xi, 1) ~= sampleNum && size(Xi, 2) == sampleNum
        X{iv} = Xi';
        summaryCount = summaryCount + 1;
        summaryParts{summaryCount} = sprintf('第 %d 个视图已在内存中由 %d x %d 转为 %d x %d', ...
            iv, size(Xi, 1), size(Xi, 2), size(X{iv}, 1), size(X{iv}, 2));
    end
end

if summaryCount == 0
    summaryText = '未触发视图方向修正';
else
    summaryText = strjoin(summaryParts(1:summaryCount), '；');
end
end


function [X, Y, summaryText] = apply_dataset_adjustment(X, Y, datasetFile, adjustmentOptions, orientationSummary)
% APPLY_DATASET_ADJUSTMENT 在内存中执行 Caltech256 适配，不改动原始 .mat 文件。

summaryParts = cell(5, 1);
summaryCount = 0;
if ~isempty(orientationSummary)
    summaryCount = summaryCount + 1;
    summaryParts{summaryCount} = orientationSummary;
end

currentLabels = unique(Y);
if adjustmentOptions.removeClutterClass && contains(lower(datasetFile), 'caltech256') ...
        && numel(currentLabels) == 257
    clutterLabel = max(currentLabels);
    keepMask = (Y ~= clutterLabel);
    if ~any(keepMask)
        error('移除疑似 clutter 类别后无剩余样本，请检查标签设置。');
    end
    X = subset_multiview_data(X, keepMask);
    Y = Y(keepMask);
    summaryCount = summaryCount + 1;
    summaryParts{summaryCount} = sprintf('已在内存中移除疑似 clutter 类别(label=%d)', clutterLabel);
end

if ~isempty(adjustmentOptions.selectedLabels)
    selectedLabels = unique(double(adjustmentOptions.selectedLabels(:)));
    keepMask = ismember(Y, selectedLabels);
    if ~any(keepMask)
        error('selectedLabels 未命中任何样本，请检查 adjustmentOptions.selectedLabels。');
    end
    X = subset_multiview_data(X, keepMask);
    Y = Y(keepMask);
    summaryCount = summaryCount + 1;
    summaryParts{summaryCount} = sprintf('已在内存中筛选指定标签，共保留 %d 个标签', numel(unique(Y)));
end

if ~isempty(adjustmentOptions.maxSamplesPerClass)
    [X, Y, sampledClassCount] = limit_samples_per_class(X, Y, adjustmentOptions.maxSamplesPerClass, ...
        adjustmentOptions.adjustmentSeed);
    summaryCount = summaryCount + 1;
    summaryParts{summaryCount} = sprintf('已按类限样，每类最多保留 %d 个样本，当前类别数=%d', ...
        adjustmentOptions.maxSamplesPerClass, sampledClassCount);
end

if adjustmentOptions.remapLabels
    Y = remap_labels_to_consecutive(Y);
    summaryCount = summaryCount + 1;
    summaryParts{summaryCount} = '已将标签重映射为从 1 开始的连续整数';
end

if summaryCount == 0
    summaryText = '未做内存级数据调整';
else
    summaryText = strjoin(summaryParts(1:summaryCount), '；');
end
end


function X = subset_multiview_data(X, mask)
% SUBSET_MULTIVIEW_DATA 用统一逻辑对子集样本做视图筛选。

mask = logical(mask(:));
for iv = 1:numel(X)
    X{iv} = X{iv}(mask, :);
end
end


function [X, Y, classCount] = limit_samples_per_class(X, Y, maxSamplesPerClass, adjustmentSeed)
% LIMIT_SAMPLES_PER_CLASS 在内存中对每类样本数做上限裁剪。

if ~isscalar(maxSamplesPerClass) || ~isnumeric(maxSamplesPerClass) || ...
        ~isfinite(maxSamplesPerClass) || maxSamplesPerClass < 1 || ...
        maxSamplesPerClass ~= floor(maxSamplesPerClass)
    error('adjustmentOptions.maxSamplesPerClass 必须是正整数。');
end
if ~isscalar(adjustmentSeed) || ~isnumeric(adjustmentSeed) || ...
        ~isfinite(adjustmentSeed) || adjustmentSeed ~= floor(adjustmentSeed)
    error('adjustmentOptions.adjustmentSeed 必须是有限整数。');
end

prevRngState = rng;
cleanupObj = onCleanup(@() rng(prevRngState));
rng(adjustmentSeed, 'twister');

classLabels = unique(Y);
selectedIndexCell = cell(numel(classLabels), 1);
for ic = 1:numel(classLabels)
    classIndex = find(Y == classLabels(ic));
    if numel(classIndex) > maxSamplesPerClass
        sampledPos = randperm(numel(classIndex), maxSamplesPerClass);
        classIndex = classIndex(sort(sampledPos));
    end
    selectedIndexCell{ic} = classIndex(:);
end
selectedIndex = vertcat(selectedIndexCell{:});

clear cleanupObj;
selectedIndex = sort(selectedIndex);

keepMask = false(numel(Y), 1);
keepMask(selectedIndex) = true;
X = subset_multiview_data(X, keepMask);
Y = Y(keepMask);
classCount = numel(unique(Y));
end


function Y = remap_labels_to_consecutive(Y)
% REMAP_LABELS_TO_CONSECUTIVE 将标签压缩为从 1 开始的连续整数。

uniqueLabels = unique(Y);
remappedY = zeros(size(Y));
for i = 1:numel(uniqueLabels)
    remappedY(Y == uniqueLabels(i)) = i;
end
Y = remappedY;
end


function validate_search_config(datasetFile, betaList, lambdaList, rhoList, tauSList, ...
    epsilonValue, repeatNum, baseSeed, labelFieldCandidates, adjustmentOptions)
% VALIDATE_SEARCH_CONFIG 校验网格搜索配置参数。

if ~(ischar(datasetFile) || (isstring(datasetFile) && isscalar(datasetFile)))
    error('datasetFile 必须是字符向量或字符串标量。');
end
if isempty(betaList) || ~isnumeric(betaList) || any(~isfinite(betaList)) || any(betaList <= 0)
    error('betaList 必须是非空正数向量。');
end
if isempty(lambdaList) || ~isnumeric(lambdaList) || any(~isfinite(lambdaList)) || any(lambdaList <= 0)
    error('lambdaList 必须是非空正数向量。');
end
if isempty(rhoList) || ~isnumeric(rhoList) || any(~isfinite(rhoList)) || any(rhoList < 0) || any(rhoList > 1)
    error('rhoList 必须是位于 [0, 1] 的非空数值向量。');
end
if isempty(tauSList) || ~isnumeric(tauSList) || any(~isfinite(tauSList)) || any(tauSList <= 0)
    error('tauSList 必须是非空正数向量。');
end
if ~isscalar(epsilonValue) || ~isnumeric(epsilonValue) || ~isfinite(epsilonValue) || epsilonValue <= 0
    error('epsilonValue 必须是有限正数。');
end
if ~isscalar(repeatNum) || repeatNum < 1 || repeatNum ~= floor(repeatNum)
    error('repeatNum 必须是大于等于 1 的整数。');
end
if ~isscalar(baseSeed) || ~isfinite(baseSeed) || baseSeed ~= floor(baseSeed)
    error('baseSeed 必须是有限整数。');
end
if ~iscell(labelFieldCandidates) || isempty(labelFieldCandidates)
    error('labelFieldCandidates 必须是非空 cell 数组。');
end
if ~isstruct(adjustmentOptions)
    error('adjustmentOptions 必须是结构体。');
end
end


function labelField = resolve_label_field(datasetStruct, labelFieldCandidates, datasetFile)
% RESOLVE_LABEL_FIELD 自动解析标签字段名。

for i = 1:numel(labelFieldCandidates)
    candidate = labelFieldCandidates{i};
    if isfield(datasetStruct, candidate)
        value = datasetStruct.(candidate);
        if isnumeric(value) && isvector(value) && ~isempty(value)
            labelField = candidate;
            return;
        end
    end
end

fieldList = fieldnames(datasetStruct);
if isfield(datasetStruct, 'X') && iscell(datasetStruct.X) && ~isempty(datasetStruct.X)
    sampleNum = max(size(datasetStruct.X{1}));
    for i = 1:numel(fieldList)
        currentField = fieldList{i};
        if strcmp(currentField, 'X')
            continue;
        end
        value = datasetStruct.(currentField);
        if isnumeric(value) && isvector(value) && numel(value) == sampleNum && ~isempty(value)
            labelField = currentField;
            return;
        end
    end
end

error(['数据集 %s 中未找到可识别的标签字段。', ...
    '请检查变量名，或将字段名加入 labelFieldCandidates。'], datasetFile);
end


function [X, Y] = validate_multiview_data(X, Y, datasetFile)
% VALIDATE_MULTIVIEW_DATA 对多视图数据执行基本输入检查。

if isempty(Y) || ~isnumeric(Y) || ~isvector(Y)
    error('数据集 %s 中的标签必须是非空数值向量。', datasetFile);
end
if any(~isfinite(Y))
    error('数据集 %s 中的标签包含 NaN 或 Inf，请先检查数据。', datasetFile);
end

sampleNum = numel(Y);
for iv = 1:numel(X)
    Xi = X{iv};
    if isempty(Xi)
        error('数据集 %s 的第 %d 个视图为空。', datasetFile, iv);
    end
    if ~(isnumeric(Xi) || islogical(Xi))
        error('数据集 %s 的第 %d 个视图必须是数值矩阵。', datasetFile, iv);
    end
    if size(Xi, 1) ~= sampleNum
        error(['数据集 %s 的第 %d 个视图样本数与标签长度不一致：', ...
            'size(X{%d},1)=%d, numel(Y)=%d。'], ...
            datasetFile, iv, iv, size(Xi, 1), sampleNum);
    end
    if any(~isfinite(Xi(:)))
        error('数据集 %s 的第 %d 个视图包含 NaN 或 Inf，请先检查数据。', datasetFile, iv);
    end
    X{iv} = double(Xi);
end

if numel(unique(Y)) < 2
    error('数据集 %s 的标签类别数小于 2，无法进行聚类评价。', datasetFile);
end
end


function record = create_empty_record(metricNames, viewNum)
% CREATE_EMPTY_RECORD 创建单次搜索结果结构体模板。

record = struct( ...
    'datasetFile', '', ...
    'datasetName', '', ...
    'labelField', '', ...
    'originalSampleNum', NaN, ...
    'originalClassNum', NaN, ...
    'sampleNum', NaN, ...
    'viewNum', viewNum, ...
    'classNum', NaN, ...
    'adjustmentSummary', '', ...
    'beta', NaN, ...
    'lambda', NaN, ...
    'rho', NaN, ...
    'tauS', NaN, ...
    'epsilonValue', NaN, ...
    'repeatId', NaN, ...
    'seed', NaN, ...
    'targetView', NaN, ...
    'iterNum', NaN, ...
    'finalObj', NaN, ...
    'neighborTime', NaN, ...
    'elapsedTime', NaN, ...
    'status', '', ...
    'errorMessage', '', ...
    'qualityScores', nan(1, viewNum), ...
    'qualityScoresText', '', ...
    'alignmentErrors', nan(1, viewNum), ...
    'alignmentErrorsText', '', ...
    'jointScores', nan(1, viewNum), ...
    'jointScoresText', '', ...
    'viewWeights', nan(1, viewNum), ...
    'viewWeightsText', '');

for i = 1:numel(metricNames)
    record.(metricNames{i}) = NaN;
    record.([metricNames{i}, 'Std']) = NaN;
end
for iv = 1:viewNum
    record.(sprintf('quality_v%d', iv)) = NaN;
    record.(sprintf('align_v%d', iv)) = NaN;
    record.(sprintf('score_v%d', iv)) = NaN;
    record.(sprintf('weight_v%d', iv)) = NaN;
end
end


function record = build_search_record(datasetInfo, metricNames, viewNum, beta, lambda, rho, tauS, ...
    epsilonValue, repeatId, seed, targetView, iterNum, obj, neighborTime, elapsedTime, ...
    qualityScores, alignmentErrors, jointScores, viewWeights, metricMean, metricStd, ...
    status, errorMessage)
% BUILD_SEARCH_RECORD 组织单次搜索结果。

record = create_empty_record(metricNames, viewNum);
record.datasetFile = datasetInfo.datasetFile;
record.datasetName = datasetInfo.datasetName;
record.labelField = datasetInfo.labelField;
record.originalSampleNum = datasetInfo.originalSampleNum;
record.originalClassNum = datasetInfo.originalClassNum;
record.sampleNum = datasetInfo.sampleNum;
record.classNum = datasetInfo.classNum;
record.adjustmentSummary = datasetInfo.adjustmentSummary;
record.beta = beta;
record.lambda = lambda;
record.rho = rho;
record.tauS = tauS;
record.epsilonValue = epsilonValue;
record.repeatId = repeatId;
record.seed = seed;
record.targetView = targetView;
record.iterNum = iterNum;
if isempty(obj)
    record.finalObj = NaN;
else
    record.finalObj = obj(end);
end
record.neighborTime = neighborTime;
record.elapsedTime = elapsedTime;
record.status = status;
record.errorMessage = errorMessage;
record.qualityScores = qualityScores(:)';
record.qualityScoresText = vector_to_text(qualityScores);
record.alignmentErrors = alignmentErrors(:)';
record.alignmentErrorsText = vector_to_text(alignmentErrors);
record.jointScores = jointScores(:)';
record.jointScoresText = vector_to_text(jointScores);
record.viewWeights = viewWeights(:)';
record.viewWeightsText = vector_to_text(viewWeights);

for i = 1:numel(metricNames)
    record.(metricNames{i}) = metricMean(i);
    record.([metricNames{i}, 'Std']) = metricStd(i);
end
for iv = 1:viewNum
    record.(sprintf('quality_v%d', iv)) = qualityScores(iv);
    record.(sprintf('align_v%d', iv)) = alignmentErrors(iv);
    record.(sprintf('score_v%d', iv)) = jointScores(iv);
    record.(sprintf('weight_v%d', iv)) = viewWeights(iv);
end
end


function combo = create_empty_combo_summary(viewNum)
% CREATE_EMPTY_COMBO_SUMMARY 创建参数组合汇总模板。

combo = struct( ...
    'beta', NaN, ...
    'lambda', NaN, ...
    'rho', NaN, ...
    'tauS', NaN, ...
    'bestACC', NaN, ...
    'meanACC', NaN, ...
    'stdACC', NaN, ...
    'bestTime', NaN, ...
    'meanTime', NaN, ...
    'meanAlignmentErrors', nan(1, viewNum), ...
    'meanAlignmentErrorsText', '', ...
    'meanWeights', nan(1, viewNum), ...
    'meanWeightsText', '');

for iv = 1:viewNum
    combo.(sprintf('mean_align_v%d', iv)) = NaN;
    combo.(sprintf('mean_weight_v%d', iv)) = NaN;
end
end


function combo = build_combo_summary(combo, beta, lambda, rho, tauS, comboACC, comboTime, ...
    comboWeights, comboAlignmentErrors)
% BUILD_COMBO_SUMMARY 组织参数组合汇总结果。

[bestACC, meanACC, stdACC] = compute_valid_statistics(comboACC, 'max');
[bestTime, meanTime] = compute_time_statistics(comboTime);
meanWeights = compute_mean_vector(comboWeights);
meanAlignmentErrors = compute_mean_vector(comboAlignmentErrors);

combo.beta = beta;
combo.lambda = lambda;
combo.rho = rho;
combo.tauS = tauS;
combo.bestACC = bestACC;
combo.meanACC = meanACC;
combo.stdACC = stdACC;
combo.bestTime = bestTime;
combo.meanTime = meanTime;
combo.meanAlignmentErrors = meanAlignmentErrors;
combo.meanAlignmentErrorsText = vector_to_text(meanAlignmentErrors);
combo.meanWeights = meanWeights;
combo.meanWeightsText = vector_to_text(meanWeights);
for iv = 1:numel(meanWeights)
    combo.(sprintf('mean_align_v%d', iv)) = meanAlignmentErrors(iv);
    combo.(sprintf('mean_weight_v%d', iv)) = meanWeights(iv);
end
end


function [bestValue, meanValue, stdValue] = compute_valid_statistics(values, mode)
% COMPUTE_VALID_STATISTICS 仅基于有限值计算统计量。

validValues = values(isfinite(values));
if isempty(validValues)
    bestValue = NaN;
    meanValue = NaN;
    stdValue = NaN;
    return;
end

if strcmp(mode, 'max')
    bestValue = max(validValues);
elseif strcmp(mode, 'min')
    bestValue = min(validValues);
else
    error('mode 必须为 ''max'' 或 ''min''。');
end
meanValue = mean(validValues);
stdValue = std(validValues, 1);
end


function [bestTime, meanTime] = compute_time_statistics(values)
% COMPUTE_TIME_STATISTICS 计算有限时间值的最小值和均值。

[bestTime, meanTime] = compute_valid_statistics(values, 'min');
end


function meanVector = compute_mean_vector(values)
% COMPUTE_MEAN_VECTOR 对每列有限值求均值。

if isempty(values)
    meanVector = [];
    return;
end

meanVector = nan(1, size(values, 2));
for iv = 1:size(values, 2)
    currentValues = values(:, iv);
    currentValues = currentValues(isfinite(currentValues));
    if ~isempty(currentValues)
        meanVector(iv) = mean(currentValues);
    end
end
end


function text = vector_to_text(values)
% VECTOR_TO_TEXT 将向量格式化为日志字符串。

if isempty(values) || all(~isfinite(values(:)))
    text = '[NaN]';
    return;
end

parts = cell(numel(values), 1);
for i = 1:numel(values)
    if isfinite(values(i))
        parts{i} = sprintf('v%d=%.6f', i, values(i));
    else
        parts{i} = sprintf('v%d=NaN', i);
    end
end
text = strjoin(parts, ', ');
end
