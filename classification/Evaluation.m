function [Acc,Acc1,accuracy,precision,recall,specificity,F1score,kappa,MCC,Yy] = Evaluation(y,Y,n)
%用来评价分类结果
%accuracy 准确率
%precision 精度
%recall 召回率
%F1core F1值
%kappa kappa值
%y 预测结果          向量
%Y 真实结果  test_y  矩阵
%n 分类种类数
%Acc
po=0;pe=0;
for i=1:n
    YY=find(Y(:,i)==1);
    for j=1:n
        yy=length(find(y(YY)==j));
        acc(i,j)=yy;
        if length(YY)==0
            acc1(i,j)=0;
        else
            acc1(i,j)=yy/length(YY);
        end
    end
    Yy(i)=length(YY);
end
Acc=acc;
Acc1=acc1;
    for ii=1:size(Y,1)
        Y1(ii)=find(Y(ii,:)==1);
    end
    Y1=Y1';
% Y1=Y;
for k=1:n
    TP=length(find(y==k & Y1==k));
    FP=length(find(y==k & Y1~=k));
    TN=length(find(y~=k & Y1~=k));
    FN=length(find(y~=k & Y1==k));

    accuracy(k)=calindex(TP,Yy(k));
    precision(k)=calindex(TP,TP+FP);
    recall(k)=calindex(TP,TP+FN);%Sensitivity
    specificity(k)=calindex(TN,FP+TN);%Specificity
    F1score(k)=calindex(2*precision(k)*recall(k),precision(k)+recall(k));
    MCC(k)=calindex(TP*TN-FP*FN,sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)));
    
    po=po+TP;
%     pe=pe+acc(k,k)*Yy(k)/length(Y)^2;
    pe=pe+sum(Acc1(k,:))*sum(Acc1(:,k));
    clear TP
    clear FP
    clear TN
    clear FN
end
Yy=Yy/length(y);
pe=pe/sum(sum(Acc1))^2;
po=po/length(Y);
kappa=(po-pe)/(1-pe);
end

function c=calindex(a,b)
    if b==0
        c=0;
    else
        c=a/b;
    end
end