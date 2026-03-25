function [Train_x, Test_x ]=pre_zca(train_x,test_x)
%白化  是训练更易收敛
train_x = bsxfun(@rdivide, bsxfun(@minus, train_x, mean(train_x,2)), sqrt(var(train_x,[],2)+10));%bsxfun对矩阵中每个元素进行操作  var方差
%公式为：  train_x中每个元素减去该样本对应均值，在右除train_x的std+10
C = cov(train_x);%协方差矩阵   X=A*Az转置*1/|A|
M = mean(train_x);
[V,D] = eig(C);%特征值V与特征向量D
P = V * diag(sqrt(1./(diag(D) + 1e2))) * V';%用奇异值简化矩阵  V* 奇异值 *V'    diag取对角线元素
Train_x = bsxfun(@minus, train_x, M) * P;
 
test_x = bsxfun(@rdivide, bsxfun(@minus, test_x, mean(test_x,2)), sqrt(var(test_x,[],2)+10));
Test_x = bsxfun(@minus, test_x, M) * P;