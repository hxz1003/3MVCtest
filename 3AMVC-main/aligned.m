function [S, T, fusionInfo] = aligned(Z, c, target_view, fusionOptions)
% ALIGNED 执行跨视图锚图对齐与融合。
% 功能简介：
% 先将各非基准视图的锚图对齐到基准视图，再根据配置执行等权融合、
% 质量感知加权融合或质量-对齐联合加权融合。
%
% 输入参数说明：
%   Z            : cell，长度为 v。第 i 个元素为 m_i x n 的锚图。
%   c            : 标量，对齐阶段 DSPFP 的权衡参数。
%   target_view  : 标量，基准视图索引。
%   fusionOptions: 结构体，可选，支持字段：
%                  - mode: 'uniform'、'quality_weighted'、
%                          'quality_alignment_weighted'
%                  - useQualityWeight
%                  - useQualityAlignmentWeight
%                  - qualityScores
%                  - tauQ
%                  - rho
%                  - tauS
%                  - epsilon
%
% 输出参数说明：
%   S          : m_b x n 的融合锚图。
%   T          : cell，长度为 v。第 i 个元素为对齐矩阵。
%   fusionInfo : 结构体，包含融合模式、质量分数、对齐残差、联合得分和视图权重。
%
% 维度说明：
%   若基准视图锚点数为 m_b，则 T{i} 为 m_b x m_i，S 为 m_b x n。
%
% 注意事项：
% 1. 若不传 fusionOptions，本函数保持原始等权融合行为。
% 2. 质量分数越小表示该视图越可靠；对齐残差越小表示该视图越易与基准视图融合。
%
% See also algo_qp, DSPFP

numview = length(Z);
if nargin < 4 || isempty(fusionOptions)
    fusionOptions = struct();
end

validate_aligned_inputs(Z, c, target_view, numview);
fusionInfo = parse_fusion_options(fusionOptions, numview, target_view);

baselineGraph = Z{target_view};
baselineStructure = baselineGraph * baselineGraph';
alignedGraphs = cell(numview, 1);
alignedStructures = cell(numview, 1);
T = cell(numview, 1);

T{target_view} = eye(size(baselineGraph, 1));
alignedGraphs{target_view} = baselineGraph;
alignedStructures{target_view} = baselineStructure;

for nv = 1:numview
    if nv ~= target_view
        currentStructure = Z{nv} * Z{nv}';
        K = baselineGraph * Z{nv}';
        T{nv} = DSPFP(baselineStructure, currentStructure, K, c);
        alignedGraphs{nv} = T{nv} * Z{nv};
        alignedStructures{nv} = T{nv} * currentStructure * T{nv}';
    end
end

fusionInfo = finalize_fusion_info(fusionInfo, alignedGraphs, alignedStructures, ...
    baselineGraph, baselineStructure, c);

S = zeros(size(baselineGraph));
for nv = 1:numview
    S = S + fusionInfo.weights(nv) * alignedGraphs{nv};
end
end


function validate_aligned_inputs(Z, c, target_view, numview)
% VALIDATE_ALIGNED_INPUTS 检查对齐阶段关键输入。

if ~iscell(Z) || isempty(Z)
    error('输入 Z 必须是非空 cell 数组。');
end
if ~isscalar(c) || ~isnumeric(c) || ~isfinite(c) || c <= 0
    error('参数 c 必须是有限正数。');
end
if ~isscalar(target_view) || target_view < 1 || target_view > numview || target_view ~= floor(target_view)
    error('target_view 必须是位于 [1, %d] 范围内的整数。', numview);
end

sampleNum = size(Z{target_view}, 2);
for iv = 1:numview
    if isempty(Z{iv}) || ~isnumeric(Z{iv})
        error('Z{%d} 必须是非空数值矩阵。', iv);
    end
    if size(Z{iv}, 2) ~= sampleNum
        error('所有视图锚图的样本数必须一致。Z{%d} 的列数与基准视图不一致。', iv);
    end
    if any(~isfinite(Z{iv}(:)))
        error('Z{%d} 中包含 NaN 或 Inf，无法执行对齐。', iv);
    end
end
end


function fusionInfo = parse_fusion_options(fusionOptions, numview, target_view)
% PARSE_FUSION_OPTIONS 解析并生成融合配置。

fusionInfo = struct();
fusionInfo.mode = 'uniform';
fusionInfo.targetView = target_view;
fusionInfo.qualityScores = nan(numview, 1);
fusionInfo.alignmentErrors = zeros(numview, 1);
fusionInfo.normalizedQualityScores = nan(numview, 1);
fusionInfo.normalizedAlignmentErrors = nan(numview, 1);
fusionInfo.jointScores = nan(numview, 1);
fusionInfo.weights = ones(numview, 1) / numview;
fusionInfo.tauQ = 1;
fusionInfo.rho = 0.5;
fusionInfo.tauS = 1;
fusionInfo.epsilon = 1e-8;

if ~isstruct(fusionOptions)
    error('fusionOptions 必须是结构体。');
end

if isfield(fusionOptions, 'mode') && ~isempty(fusionOptions.mode)
    if ischar(fusionOptions.mode)
        fusionInfo.mode = fusionOptions.mode;
    elseif isstring(fusionOptions.mode) && isscalar(fusionOptions.mode)
        fusionInfo.mode = char(fusionOptions.mode);
    else
        error('fusionOptions.mode 必须是字符向量或字符串标量。');
    end
elseif isfield(fusionOptions, 'useQualityAlignmentWeight') && logical(fusionOptions.useQualityAlignmentWeight)
    fusionInfo.mode = 'quality_alignment_weighted';
elseif isfield(fusionOptions, 'useQualityWeight') && logical(fusionOptions.useQualityWeight)
    fusionInfo.mode = 'quality_weighted';
end

validModes = {'uniform', 'quality_weighted', 'quality_alignment_weighted'};
if ~any(strcmp(fusionInfo.mode, validModes))
    error('fusionOptions.mode 必须是 uniform、quality_weighted 或 quality_alignment_weighted。');
end

if isfield(fusionOptions, 'tauQ') && ~isempty(fusionOptions.tauQ)
    validate_positive_scalar(fusionOptions.tauQ, 'fusionOptions.tauQ');
    fusionInfo.tauQ = double(fusionOptions.tauQ);
end
if isfield(fusionOptions, 'rho') && ~isempty(fusionOptions.rho)
    rho = double(fusionOptions.rho);
    if ~isscalar(rho) || ~isfinite(rho) || rho < 0 || rho > 1
        error('fusionOptions.rho 必须位于 [0, 1]。');
    end
    fusionInfo.rho = rho;
end
if isfield(fusionOptions, 'tauS') && ~isempty(fusionOptions.tauS)
    validate_positive_scalar(fusionOptions.tauS, 'fusionOptions.tauS');
    fusionInfo.tauS = double(fusionOptions.tauS);
end
if isfield(fusionOptions, 'epsilon') && ~isempty(fusionOptions.epsilon)
    validate_positive_scalar(fusionOptions.epsilon, 'fusionOptions.epsilon');
    fusionInfo.epsilon = double(fusionOptions.epsilon);
end

if strcmp(fusionInfo.mode, 'quality_weighted') || strcmp(fusionInfo.mode, 'quality_alignment_weighted')
    if ~isfield(fusionOptions, 'qualityScores') || isempty(fusionOptions.qualityScores)
        error('启用加权融合时，必须提供 fusionOptions.qualityScores。');
    end
    qualityScores = double(fusionOptions.qualityScores(:));
    if numel(qualityScores) ~= numview
        error('qualityScores 的长度必须等于视图数 %d。', numview);
    end
    if any(~isfinite(qualityScores))
        error('qualityScores 中包含 NaN 或 Inf，无法计算融合权重。');
    end
    fusionInfo.qualityScores = qualityScores;
end
end


function validate_positive_scalar(value, fieldName)
% VALIDATE_POSITIVE_SCALAR 检查正标量参数。

if ~isscalar(value) || ~isnumeric(value) || ~isfinite(value) || value <= 0
    error('%s 必须是有限正数。', fieldName);
end
end


function fusionInfo = finalize_fusion_info(fusionInfo, alignedGraphs, alignedStructures, ...
    baselineGraph, baselineStructure, lambda)
% FINALIZE_FUSION_INFO 根据融合模式计算最终权重。

switch fusionInfo.mode
    case 'uniform'
        fusionInfo.weights = ones(numel(alignedGraphs), 1) / numel(alignedGraphs);
        fusionInfo.alignmentErrors = compute_alignment_errors(alignedGraphs, alignedStructures, ...
            baselineGraph, baselineStructure, lambda, fusionInfo.targetView, fusionInfo.epsilon);
        fusionInfo.normalizedQualityScores = nan(size(fusionInfo.qualityScores));
        fusionInfo.normalizedAlignmentErrors = normalize_scores(fusionInfo.alignmentErrors, fusionInfo.epsilon);
        fusionInfo.jointScores = nan(size(fusionInfo.qualityScores));

    case 'quality_weighted'
        fusionInfo.weights = compute_softmax_weights(fusionInfo.qualityScores, fusionInfo.tauQ);
        fusionInfo.alignmentErrors = compute_alignment_errors(alignedGraphs, alignedStructures, ...
            baselineGraph, baselineStructure, lambda, fusionInfo.targetView, fusionInfo.epsilon);
        fusionInfo.normalizedQualityScores = normalize_scores(fusionInfo.qualityScores, fusionInfo.epsilon);
        fusionInfo.normalizedAlignmentErrors = normalize_scores(fusionInfo.alignmentErrors, fusionInfo.epsilon);
        fusionInfo.jointScores = nan(size(fusionInfo.qualityScores));

    case 'quality_alignment_weighted'
        fusionInfo.alignmentErrors = compute_alignment_errors(alignedGraphs, alignedStructures, ...
            baselineGraph, baselineStructure, lambda, fusionInfo.targetView, fusionInfo.epsilon);
        fusionInfo.normalizedQualityScores = normalize_scores(fusionInfo.qualityScores, fusionInfo.epsilon);
        fusionInfo.normalizedAlignmentErrors = normalize_scores(fusionInfo.alignmentErrors, fusionInfo.epsilon);
        fusionInfo.jointScores = fusionInfo.rho * fusionInfo.normalizedQualityScores + ...
            (1 - fusionInfo.rho) * fusionInfo.normalizedAlignmentErrors;
        fusionInfo.weights = compute_softmax_weights(fusionInfo.jointScores, fusionInfo.tauS);
end
end


function alignmentErrors = compute_alignment_errors(alignedGraphs, alignedStructures, ...
    baselineGraph, baselineStructure, lambda, targetView, epsilonValue)
% COMPUTE_ALIGNMENT_ERRORS 计算各视图相对基准视图的对齐残差。

numview = numel(alignedGraphs);
alignmentErrors = zeros(numview, 1);
featureDen = max(norm(baselineGraph, 'fro')^2, epsilonValue);
structureDen = max(norm(baselineStructure, 'fro')^2, epsilonValue);

for iv = 1:numview
    if iv == targetView
        alignmentErrors(iv) = 0;
    else
        featureResidual = norm(baselineGraph - alignedGraphs{iv}, 'fro')^2 / featureDen;
        structureResidual = norm(baselineStructure - alignedStructures{iv}, 'fro')^2 / structureDen;
        alignmentErrors(iv) = featureResidual + lambda * structureResidual;
    end
end
end


function normalizedScores = normalize_scores(scores, epsilonValue)
% NORMALIZE_SCORES 按均值对向量做数值稳定归一化。

scores = scores(:);
denominator = mean(scores) + epsilonValue;
normalizedScores = scores / denominator;
end


function weights = compute_softmax_weights(scores, temperature)
% COMPUTE_SOFTMAX_WEIGHTS 根据给定分数计算 softmax 权重。

logits = -scores(:) ./ temperature;
logits = logits - max(logits);
weights = exp(logits);
weightSum = sum(weights);
if ~(isfinite(weightSum) && weightSum > 0)
    error('融合权重归一化失败，请检查分数向量或温度参数。');
end
weights = weights / weightSum;
end
