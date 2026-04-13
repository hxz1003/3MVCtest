function [UU, A, Z, iter, obj, fusionInfo] = algo_qp(X, Y, theta, beta, lambda, target_view, fusionOptions)
% ALGO_QP 迭代优化多视图锚图并执行对齐融合。
% 功能简介：
% 该函数先根据各视图锚点初始化锚图，再交替更新投影矩阵 A 与锚图 Z，
% 收敛后调用 aligned 完成跨视图对齐与融合，并通过 SVD 得到聚类嵌入。
%
% 输入参数说明：
%   X            : cell，长度为 v。第 i 个视图为 n x d_i 数据矩阵。
%   Y            : n x 1 标签向量，仅用于确定样本数与类别数。
%   theta        : cell，长度为 v。第 i 个元素为 m_i x d_i 锚点矩阵。
%   beta         : 标量，锚图正则项系数。
%   lambda       : 标量，对齐阶段的权衡参数。
%   target_view  : 标量，基准视图索引。
%   fusionOptions: 结构体，可选，透传给 aligned，用于控制 uniform、
%                  quality_weighted 或 quality_alignment_weighted 融合模式。
%
% 输出参数说明：
%   UU         : n x m_b 左奇异向量矩阵。
%   A          : cell，长度为 v。每个元素为视图投影矩阵。
%   Z          : m_b x n 融合后锚图。
%   iter       : 实际迭代次数。
%   obj        : 目标函数值序列。
%   fusionInfo : 结构体，包含融合模式、质量分数、对齐残差、联合得分和视图权重。
%
% 维度说明：
%   若第 i 个视图锚点数为 m_i，则 A{i} 为 d_i x m_i，Zi{i} 为 m_i x n。
%
% 注意事项：
% 1. 若不传 fusionOptions，本函数默认保持原始等权融合行为。
% 2. 该实现沿用原项目的 parfor 更新方式，需要 Parallel Computing Toolbox。
%
% See also aligned, Neighbor

if nargin < 7 || isempty(fusionOptions)
    fusionOptions = struct();
end

validate_algo_inputs(X, Y, theta, beta, lambda, target_view);

maxIter = 50;
numview = length(X);
numsample = size(Y, 1);
M = cell(numview, 1);
Zi = cell(numview, 1);
A = cell(numview, 1);

for i = 1:numview
    A{i} = theta{i}';
    X{i} = mapstd(double(X{i})', 0, 1);
    M{i} = A{i}' * X{i};
    Zi{i} = project_anchor_graph(M{i}, size(theta{i}, 1), numsample);
end

flag = true;
iter = 0;
obj = zeros(maxIter, 1);

while flag
    iter = iter + 1;

    parfor ia = 1:numview
        if size(A{ia}, 1) < size(A{ia}, 2)
            A{ia} = X{ia} * Zi{ia}' * pinv(Zi{ia} * Zi{ia}');
        else
            C = X{ia} * Zi{ia}';
            [U, ~, V] = svd(C, 'econ');
            A{ia} = U * V';
        end
    end

    for a = 1:numview
        M{a} = (A{a}' * X{a}) / (1 + beta);
        Zi{a} = project_anchor_graph(M{a}, size(theta{a}, 1), numsample);
    end

    term1 = 0;
    term2 = 0;
    for iv = 1:numview
        term1 = term1 + norm(X{iv} - A{iv} * Zi{iv}, 'fro')^2;
        term2 = term2 + norm(Zi{iv}, 'fro')^2;
    end
    obj(iter) = term1 + beta * term2;

    if should_stop(iter, obj(iter), obj)
        obj = obj(1:iter);
        [Z, ~, fusionInfo] = aligned(Zi, lambda, target_view, fusionOptions);
        [UU, ~, ~] = svd(Z', 'econ');
        flag = false;
    end
end
end


function validate_algo_inputs(X, Y, theta, beta, lambda, target_view)
% VALIDATE_ALGO_INPUTS 检查主优化过程输入。

if ~iscell(X) || isempty(X)
    error('输入 X 必须是非空 cell 数组。');
end
if ~iscell(theta) || numel(theta) ~= numel(X)
    error('theta 必须是与 X 等长的 cell 数组。');
end
if isempty(Y) || ~isnumeric(Y) || ~isvector(Y)
    error('Y 必须是非空数值标签向量。');
end
if ~isscalar(beta) || ~isnumeric(beta) || ~isfinite(beta) || beta <= 0
    error('beta 必须是有限正数。');
end
if ~isscalar(lambda) || ~isnumeric(lambda) || ~isfinite(lambda) || lambda <= 0
    error('lambda 必须是有限正数。');
end
if ~isscalar(target_view) || target_view < 1 || target_view > numel(X) || target_view ~= floor(target_view)
    error('target_view 必须是位于 [1, %d] 范围内的整数。', numel(X));
end

sampleNum = numel(Y);
for iv = 1:numel(X)
    if isempty(X{iv}) || ~(isnumeric(X{iv}) || islogical(X{iv}))
        error('X{%d} 必须是非空数值矩阵。', iv);
    end
    if size(X{iv}, 1) ~= sampleNum
        error('X{%d} 的样本数与标签长度不一致。', iv);
    end
    if isempty(theta{iv}) || ~isnumeric(theta{iv}) || size(theta{iv}, 2) ~= size(X{iv}, 2)
        error('theta{%d} 的维度必须与 X{%d} 的特征维度一致。', iv, iv);
    end
end
end


function Z = project_anchor_graph(M, anchorNum, sampleNum)
% PROJECT_ANCHOR_GRAPH 将每个样本投影到概率单纯形上。

Z = zeros(anchorNum, sampleNum);
for ii = 1:sampleNum
    idx = 1:anchorNum;
    Z(idx, ii) = EProjSimplex_new(M(idx, ii));
end
end


function stopFlag = should_stop(iter, currentObj, obj)
% SHOULD_STOP 判断优化是否终止。

stopFlag = false;
if iter <= 9
    return;
end

prevObj = obj(iter - 1);
relativeGap = abs((prevObj - currentObj) / max(abs(prevObj), eps));
if relativeGap < 1e-6 || iter >= numel(obj) || currentObj < 1e-10
    stopFlag = true;
end
end
