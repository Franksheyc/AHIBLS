function [statisticosa,icc_score]=osa_regression1(k,block,ahixx,ahix,S,S1,osacha,leng,bmi,sleepeff,label)
%测试osa的回归模型
y=[];s=[];Y20=[];YY20=[];
for i=1:k
    yy20=ahixx{i,block};y20=ahix{i,block};
    ss=S1{i};ss1=S{i};
    [~, ~, y1 ,MAE, RMSE] = osares(leng,ss,ss1,yy20,y20,osacha,bmi,sleepeff,label);
    s=[s;S{i}];
    y=[y;y1]; 
    Y20=[Y20;y20];
    YY20=[YY20;yy20];
end
y=[y s];
y=sortrows(y,2);
y=y(:,1);
y(y<0)=0;

% y0=intersect( find(osacha>quantile(osacha,0.05)),find(osacha<quantile(osacha,0.95)) );
% y=y(y0);
% osacha=osacha(y0);

[statisticosa,yl,osal]=osasta(y,osacha);
[icc_score, ~, ~, F, ~, ~, p] = ICC([yl osal] , '1-1');
MAE=mean(abs(y-osacha));
end

function [linearModel, x, y ,MAE, RMSE] = ahical(leng,ss,ss1,yy20,y20,osacha,bmi,sleepeff)
[train_x,train_y]=geosada(leng,ss,yy20,osacha,bmi,sleepeff);
[test_x,test_y]=geosada(leng,ss1,y20,osacha,bmi,sleepeff);
linearModel = fitlm(train_x, train_y,'linear', 'RobustOpts', 'logistic');%'linear'     'logistic'
x=predict(linearModel,train_x);
y=predict(linearModel,test_x);
MAE=mean(test_y-y);
RMSE=sqrt(abs(sum(test_y-y).^2));
end

function [osadatr,osatr]=geosada(leng,ss,yy20,osacha,bmi,sleepeff,label)
k=0;lengss=leng(ss);lenm=mean(leng);
for i=1:length(ss)
k(i)=sum(lengss(1:i));
end
k=[0 k];
for i=1:length(ss)
da=yy20(k(i)+1:k(i+1));
% da=da(1:min(2*60*7,length(da)));%nc文章中的前7小时
dal=rem(1:length(da),2);
dal=dal(1:find(dal==0,1,'last'));
da1=da(find(dal==1))+da(find(dal==0));
osadatr(i,1)=length(find(da==1))/length(da);
osadatr(i,2)=length(find(da==2))/length(da);
osadatr(i,3)=length(find(diff(da)~=0))/length(da);
osadatr(i,4)=length(find(diff(da)==0))/length(da);
% osadatr(i,5)=log(length(find(da==1)));
% osadatr(i,6)=log(length(find(da==2)));
% osadatr(i,6)=length(find(da1==2))/length(da1);
% osadatr(i,6)=length(find(da1==3))/length(da1);
% osadatr(i,7)=length(find(da1==4))/length(da1);
% osadatr(i,5)=(length(da)-lenm)/lenm;
end
% if mean(sleepeff)<1
%    sleepeff=sleepeff*100;
% end
osadatr=[osadatr bmi(ss)>30 ];
osatr=osacha(ss);
end

% function [linearModel, x, y ,MAE, RMSE] = osares(leng,ss,ss1,yy20,y20,osacha,bmi,sleepeff)
% [train_x,train_y]=geosada(leng,ss,yy20,osacha,bmi,sleepeff);
% [test_x,test_y]=geosada(leng,ss1,y20,osacha,bmi,sleepeff);
% linearModel = fitlm(train_x, train_y,'linear', 'RobustOpts', 'logistic');%'linear'     'logistic'
% x=predict(linearModel,train_x);
% y=predict(linearModel,test_x);
% MAE=mean(test_y-y);
% RMSE=sqrt(abs(sum(test_y-y).^2));
% end
% 
% function [osadatr,osatr]=geosada(leng,ss,yy20,osacha,bmi,sleepeff)
% k=0;lengss=leng(ss);lenm=mean(leng);
% for i=1:length(ss)
% k(i)=sum(lengss(1:i));
% end
% k=[0 k];
% for i=1:length(ss)
% da=yy20(k(i)+1:k(i+1));
% % da=da(1:min(2*60*7,length(da)));%nc文章中的前7小时
% dal=rem(1:length(da),2);
% dal=dal(1:find(dal==0,1,'last'));
% da1=da(find(dal==1))+da(find(dal==0));
% osadatr(i,1)=length(find(da==1))/length(da);
% osadatr(i,2)=length(find(da==2))/length(da);
% osadatr(i,3)=length(find(diff(da)~=0))/length(da);
% osadatr(i,4)=length(find(diff(da)==0))/length(da);
% % osadatr(i,5)=log(length(find(da==1)));
% % osadatr(i,6)=log(length(find(da==2)));
% % osadatr(i,6)=length(find(da1==2))/length(da1);
% % osadatr(i,6)=length(find(da1==3))/length(da1);
% % osadatr(i,7)=length(find(da1==4))/length(da1);
% % osadatr(i,5)=(length(da)-lenm)/lenm;
% end
% % if mean(sleepeff)<1
% %    sleepeff=sleepeff*100;
% % end
% osadatr=[osadatr bmi(ss)>30 ];
% osatr=osacha(ss);
% end

function [statistic,a1,b1]=osasta(a,b)
n=[0 5 15 30];
a1=ones(size(a,1),1);
b1=ones(size(b,1),1);
for i=1:length(n)-1
    n1=n(i);n2=n(i+1);
    a1(intersect(find(a>=n1),find(a<n2)))=(i+1);
    b1(intersect(find(b>=n1),find(b<n2)))=(i+1);
end
    ac=length(find(a1==b1))/length(a1);
    [con,Acc_part_1,accuracy_1,precision_1,recall_1,specificity_1,F1score_1,kappa_1,mcc_1] = Evaluation(a1,label2lab(b1),length(n));
    statistic.ahi=n;
    statistic.acc=ac;
    statistic.recall=recall_1;
    statistic.specificity=specificity_1;
    statistic.confusion=con;
    statistic.f1=F1score_1;
    statistic.mf1=mean(F1score_1);
    statistic.kappa=kappa_1;
end