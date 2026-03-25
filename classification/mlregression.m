function [result,time]=mlregression(train_x,train_yy,test_x,test_yy,X,X1)
i=1;
tic
cmethod='NN';
if ~isempty(X)
   train_x=[train_x X{i}];
   test_x=[test_x X1{i}];
end
par = fitrnet(train_x, train_yy, 'LayerSizes', 10, 'Activations', 'relu', 'Lambda', 0, 'IterationLimit', 1000, 'Standardize', true);
y=predict(par,test_x);
yy=predict(par,train_x);
result(i).method=cmethod;
result(i).result=y;
result(i).resultr=yy;
% result(i).acc=testacc;
time(i)=toc;

%%linear
tic
cmethod='linear';
i=i+1;
if ~isempty(X)
   train_x=[train_x X{i}];
   test_x=[test_x X1{i}];
end
par = fitlm(train_x,train_yy,'linear');
y=predict(par,test_x);
yy=predict(par,train_x);
%         [yy,xx]=predict(par,train_x);
testacc=length(find(y==test_yy))/length(y);
result(i).method=cmethod;
result(i).result=y;
result(i).resultr=yy;
% result(i).acc=testacc;
time(i)=toc;

%%tree
tic
cmethod='tree';
i=i+1;
if ~isempty(X)
   train_x=[train_x X{i}];
   test_x=[test_x X1{i}];
end
par = fitrtree(train_x,train_yy,'MinLeafSize', 4);
y=predict(par,test_x);
yy=predict(par,train_x);
result(i).method=cmethod;
result(i).result=y;
result(i).resultr=yy;
% result(i).acc=testacc;
time(i)=toc;

%%% liner SVM
tic
i=i+1;
cmethod='svm_liner';
if ~isempty(X)
   train_x=[train_x X{i}];
   test_x=[test_x X1{i}];
end
responseScale = iqr(train_yy);
boxConstraint = responseScale/1.349;
epsilon = responseScale/13.49;
par = fitrsvm(train_x,train_yy,'KernelFunction', 'linear','PolynomialOrder', [],'KernelScale', 'auto','BoxConstraint', boxConstraint,'Epsilon', epsilon,'Standardize', true);
%%%
[y]=predict(par,test_x);
yy=predict(par,train_x);
% acc(i)=testacc;
result(i).method=cmethod;
result(i).result=y;
result(i).resultr=yy;
% result(i).acc=testacc;
time(i)=toc;

% %%% svm RBF
tic
i=i+1;
cmethod='svm_RBF';
if ~isempty(X)
   train_x=[train_x X{i}];
   test_x=[test_x X1{i}];
end
par = fitrsvm(train_x,train_yy,'KernelFunction', 'gaussian','PolynomialOrder', [],'KernelScale', 13,'BoxConstraint', boxConstraint,'Epsilon', epsilon,'Standardize', true);
% %%%
[y]=predict(par,test_x);
yy=predict(par,train_x);
% acc(i)=testacc;
result(i).method=cmethod;
result(i).result=y;
result(i).resultr=yy;
% result(i).acc=testacc;
time(i)=toc;

%% 二次有理高斯过程
tic
i=i+1;
cmethod='Gaussian Processes Regression, GPR';
if ~isempty(X)
   train_x=[train_x X{i}];
   test_x=[test_x X1{i}];
end
par = fitrgp(train_x, train_yy, 'BasisFunction', 'constant', 'KernelFunction', 'rationalquadratic', 'Standardize', true);
[y]=predict(par,test_x);
yy=predict(par,train_x);
% acc(i)=testacc;
result(i).method=cmethod;
result(i).result=y;
result(i).resultr=yy;
% result(i).acc=testacc;
time(i)=toc;

%%%%%RF
tic
i=i+1;
cmethod='boost tree';
if ~isempty(X)
   train_x=[train_x X{i}];
   test_x=[test_x X1{i}];
end
template = templateTree('MinLeafSize', 8);
par = fitrensemble(train_x,train_yy,'Method', 'LSBoost','NumLearningCycles', 30, 'Learners', template, 'LearnRate', 0.1);
y=predict(par,test_x);
yy=predict(par,train_x);
result(i).method=cmethod;
result(i).result=y;
result(i).resultr=yy;
% result(i).acc=testacc;
time(i)=toc;

%%%%%RF
tic
i=i+1;
cmethod='bagged tree';
if ~isempty(X)
   train_x=[train_x X{i}];
   test_x=[test_x X1{i}];
end
template = templateTree('MinLeafSize', 8);
par = fitrensemble(train_x,train_yy,'Method', 'Bag','NumLearningCycles', 30, 'Learners', template);
y=predict(par,test_x);
yy=predict(par,train_x);
result(i).method=cmethod;
result(i).result=y;
result(i).resultr=yy;
% result(i).acc=testacc;
time(i)=toc;
