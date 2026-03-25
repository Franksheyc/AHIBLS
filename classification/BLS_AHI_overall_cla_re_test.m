%%%%%%%%%%%%%%%使用全部特征，time 10s/part%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
warning off all;
format compact;
tic
Dir=['D:\E\sleep\open database\breathfeatures\features'];
Dir2=['D:\E\sleep\open database\features'];
Dir1=['D:\E\sleep\open database\breathfeatures\totalfeature'];
outdir=['D:\E\sleep\ahi_prediction_result\'];
% Dir=['F:\sleep\open database\features\',add];%cassette  telemetry
% outdir=['F:\sleep\open database\result\',add];
% Dir=['F:\sleep\open database\features\','SHHS1'];
%%%
%%%%EDF==20 edf20   EDF==78 edf78   EDF==0 其他数据集
dataset={'shhs1','shhs2','cfs','sof','mesa'};
EDF=0;
epochs = 1;
n=5;
h=5;%bls enhance窗
bn=8;%测试集中每个SS中取多少组做训练
C = 2^-24; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes  C = 2^-30
times=2;%第一次bls循环次数
normark=1;% ahi  0 origin  1 log  2 event
trans=@(x)log(x+1);
trans1=@(x)exp(x)-1;
% filename='totaldata.mat';
% if ~exist(filename)
% [breathfeatures,breathfeatures1,demofeatures,blocfeatures,osacha,osacha6,sleepeff,sleepdur,age,bmi,gender,issmoke,fb,fb0,fbe,fb1,fb01,fbe1,fbloc,fb0loc,fbeloc,leng,p1,p2,shhs2loc]=datacollection(dataset,Dir1);
% save('totaldata.mat','breathfeatures','breathfeatures1','demofeatures','blocfeatures','osacha','osacha6','sleepeff','sleepdur','age','bmi','gender','issmoke','fb','fb0','fbe','fb1','fb01','fbe1','fbloc','fb0loc','fbeloc','leng','p1','p2')
% else
% load(filename)
% end
[breathfeatures,breathfeatures1,demofeatures,~,osacha,osacha6,sleepeff,sleepdur,age,bmi,gender,issmoke,fb,fb0,fbe,fb1,fb01,fbe1,~,~,~,leng,p1,p2,shhs2loc,Mark]=datacollection(dataset,Dir1);
o60=osacha(osacha>=60);
osacha(osacha>(mean(o60)+std(o60)))=mean(o60)+std(o60);


label=ones(length(osacha),1);
label=label+floor(osacha/10);
label(label>9)=9;
lab=label2lab(label,max(label));

% label(intersect(find(osacha>=5),find(osacha<15)))=2;
% label(intersect(find(osacha>=15),find(osacha<30)))=3;
% label(find(osacha>=30))=4;
% lab=label2lab(label,4);

% label1=label;label1(label~=1)=2;
% label2=label;label2(label==2)=1;label2(label~=2)=2;
% label3=label;label3(label==3)=1;label3(label~=3)=2;
% label4=label;label4(label==4)=1;label4(label~=4)=2;
% lab1=label2lab(label1,2);
% lab2=label2lab(label2,2);
% lab3=label2lab(label3,2);
% lab4=label2lab(label4,2);
% [osacha,ps]=mapminmax(osacha',0,1);
% osacha=osacha';
osacha1=osacha;
if normark==1
    osacha=trans(osacha);
    cc=trans([2.5 10 17.5 65]);
elseif normark==2
    t=(sleepdur./(sleepeff)/60*100);
    osacha=ceil(osacha.*t);
    osacha=trans(osacha);
end
hd=size(breathfeatures,1);
if length(dataset)>1
    flods=1;
else
    flods=10;
end
K=[1:size(osacha,1)];
softmax1=@(x)exp(x)./sum(exp(x)); 

[group1,group2,group3,group4,groupc,indictg1,indictg2,indictg3,indictg4,indictgc]=groupdat(age,osacha,flods);
% ACCo_pre=cell(flods,epochs);ACC_pre=cell(flods,epochs);accuracy_pre=cell(flods,epochs);
% Precidion_pre=cell(flods,epochs);Recall_pre=cell(flods,epochs);F1SCORE_pre=cell(flods,epochs);
% F1score_pre=cell(flods,epochs);Kappa_pre=cell(flods,epochs);Mcc_pre=cell(flods,epochs);
% train_length=zeros(flods,epochs);trainingacc=cell(flods,epochs);

breathfeatures(:,42)=breathfeatures(:,42)*1.2;
% breathfeatures(:,16)=breathfeatures(:,16)*1.2;

% osacha=osacha.*sleepdur/60*7;
% 24	9	42	11	16	14	21	22	23	17	10	18
% 11	24	16	9	14	12	4	23	21	22	17	42
p1=[24 	36	 42   1  9	 16	14	21	22	 17];
p2=[24  42   29   33 34 40     6 9	10	11  14   16 17  ];%  1 7 15    17 19   5 6 
p3=[1:2 4:5 8 9:14 20:23 15:19];
p4=[24:28 33 34 37:39 41:43];

fb1=[1 1 1 1 2 1 1 1 2];%ones(1,length(p1))
fb01=[1 4 5 6];
% fb2=ones(1,size(demofeatures,2));
fb2=ones(1,length(p3));
fb02=[1 6 11];
fbe2={};

fb3=ones(1,length(p4));
fb03=[1];

fb=ones(1,length(p1));
fb0=[1 3 5 7];
fbe={};%{1; 4};


for k=1:flods
    if flods~=1
        [ss,ss1,outsub]=gettrte(group1,group2,group3,group4,groupc,indictg1,indictg2,indictg3,indictg4,indictgc,k,sum(leng));
    else
%         osacha1=osacha1(1:leng(1));
%         s1=find(osacha1<=5);
%         s2=intersect(find(osacha1<=15),find(osacha1>5));
%         s3=intersect(find(osacha1<=30),find(osacha1>15));
%         s4=find(osacha1>30);
%         rng(33)
%         l=randperm(length(s1));
%         ss1=s1(l(end-109:end));s1=setdiff(s1,ss1);
%         l=randperm(length(s2));
%         ss2=s2(l(end-201:end));s2=setdiff(s2,ss2);
%         l=randperm(length(s3));
%         ss3=s3(l(end-167:end));s3=setdiff(s3,ss3);
%         l=randperm(length(s4));
%         ss4=s4(l(end-85:end));s4=setdiff(s4,ss4);
%         ss=[s1;s2;s3;s4];
        rng(33)
        rn=randperm(leng(1));
        ss=rn(1:end-562);
        ss1=setdiff(1:length(breathfeatures),ss);
        leng(1)=leng(1)-length(ss);
    end
    marktr=getmark(ss,K);
    train_y=osacha(marktr,:); 
    train_yc=label(marktr,:);
%     train_yc1=label1(marktr,:);
%     train_yc2=label2(marktr,:);
%     train_yc3=label3(marktr,:);
%     train_yc4=label4(marktr,:);
    train_yyc=lab(marktr,:);
%     train_yyc1=lab1(marktr,:);
%     train_yyc2=lab2(marktr,:);
%     train_yyc3=lab3(marktr,:);
%     train_yyc4=lab4(marktr,:);
    train_x=breathfeatures(marktr,p1);
    train_x1=breathfeatures(marktr,p2);
    train_xc=breathfeatures(marktr,p3);%blockfeatures
    train_xc1=breathfeatures(marktr,p4);%blockfeatures    
%     train_con=bfcon(marktr);

    markte=getmark(ss1,K);
    test_y=osacha(markte,:);
    test_yc=label(markte,:);
%     test_yc1=label1(markte,:);
%     test_yc2=label2(markte,:);
%     test_yc3=label3(markte,:);
%     test_yc4=label4(markte,:);
    test_yyc=lab(markte,:);
%     test_yyc1=lab1(markte,:);
%     test_yyc2=lab2(markte,:);
%     test_yyc3=lab3(markte,:);
%     test_yyc4=lab4(markte,:);
    test_x=breathfeatures(markte,p1);
    test_x1=breathfeatures(markte,p2);
    test_xc=breathfeatures(markte,p3);
    test_xc1=breathfeatures(markte,p4);
%     test_con=bfcon(markte); 

    kk=1;
    [train_x, test_x]=pre_zca(train_x,test_x);

    study_rate=0.15; xx=zeros(length(train_x),5); yy=zeros(length(train_x),1); 
    cr1=ones(length(train_x),1);ce1=ones(length(test_x),1);
%     len0=leng(ss);
%     for bo=1:length(ss)
%         ko0(bo)=sum(len0(1:bo));
%     end
%     ko0=[0 ko0];

    if kk==1
        ten=0;par=[];pare=[];par1=[];pare1=[];
    end
    wh=[];beta11=[];x_tr=zeros(length(train_x1),5);x_te=zeros(length(test_x),5);    
    ht=1;%multibls 重复次数
    
  
    f2=[];%f2为新增特征种类数
    K0=[];
    block=0;
 
for kk=1:1%train_y1 是呼吸暂停事件的标签
    
    par1=[];pare=[];
    N1=[1 2 8];
    N2=1;
    s=0.8;C=2^-24;
    h=1;fe1=[];
    nfp=0;
    block=block+1;
    leng1=ones(1,sum(leng));

    rng(256)%56
    [trainacc,testacc,xx20,x20,yy20,y20,parc,con0,~,tn]=blsc(train_x1,train_yyc,test_x1,test_yyc,s,C,h,fb1,fb01,[],[],N1,N2,fbe,[],nfp,leng1,ss1);

% %     rng(256)%56
% %     [trainacc,testacc,xx201,x201,yy201,y201,~,con0,~,tn]=blsc(train_xc,train_yyc,test_xc,test_yyc,s,C,h,fbloc,fb0loc,[],[],N1,N2,fbeloc,[],nfp,leng1,ss1);
% % 
% %     rng(256)%56
% %     [trainacc,testacc,xx202,x202,yy202,y202,~,con0,~,tn]=blsc(train_xd,train_yyc,test_xd,test_yyc,s,C,h,fb2,fb02,[],[],N1,N2,fbe2,[],nfp,leng1,ss1);
% % 
% %     xx201=(xx201+xx202)/2;
% %     x201=(x201+x202)/2;

%     rng(256)%56
%     [trainacc,testacc,xx201,x201,yy201,y201,~,con0,~,tn]=blsc([train_xc xx20],train_yyc,[test_xc x20],test_yyc,s,C,h,fbloc,fb0loc,[],[],N1,N2,fbeloc,[],nfp,leng1,ss1);
%     
%     xx20=(xx20+xx201)/2;
%     x20=(x20+x201)/2;

    rng(256)%56
    par1=[];
    [trainacc,testacc,xx201,x201,yy201,y201,parc,con0,~,tn]=blsc([train_xc xx20],train_yyc,[test_xc x20],test_yyc,s,C,h,fb2,fb02,[],[],N1,N2,fbe,[],nfp,leng1,ss1);
    
% 
%     rng(256)%56
%     par1=[];
%     [trainacc,testacc,xx202,x202,yy202,y202,~,con0,~,tn]=blsc([train_xc1 xx20],train_yyc,[test_xc1 x20],test_yyc,s,C,h,fb3,fb03,[],[],N1,N2,fbe,[],nfp,leng1,ss1);
%     
%     xx201=(xx202*1.2+xx201*0.8)/2;
%     x201=(x202*1.2+x201*0.8)/2;

    xx20=(xx20+xx201)/2;
    x20=(x20+x201)/2;
    p=[1.05 1.3 1.2 1.05 1 1 1 1 1];
    xx20=xx20.*p;
    x20=x20.*p;
%     rng(256)
%     [trainacc,testacc,xx201,x201,yy201,y201,~,con0,~,tn]=blsc(train_x1,train_yyc1,test_x1,test_yyc1,s,C,h,fb1,fb01,[],[],N1,N2,fbe,[],nfp,leng1,ss1);
%     rng(256)
%     [trainacc,testacc,xx202,x202,yy202,y202,~,con0,~,tn]=blsc(train_x1,train_yyc2,test_x1,test_yyc2,s,C,h,fb1,fb01,[],[],N1,N2,fbe,[],nfp,leng1,ss1);
%     rng(256)
%     [trainacc,testacc,xx203,x203,yy203,y203,~,con0,~,tn]=blsc(train_x1,train_yyc3,test_x1,test_yyc3,s,C,h,fb1,fb01,[],[],N1,N2,fbe,[],nfp,leng1,ss1);
%     rng(256)
%     [trainacc,testacc,xx204,x204,yy204,y204,~,con0,~,tn]=blsc(train_x1,train_yyc4,test_x1,test_yyc4,s,C,h,fb1,fb01,[],[],N1,N2,fbe,[],nfp,leng1,ss1);
%     xx20=[xx201(:,1) xx202(:,1) xx203(:,1) xx204(:,1)];
%     x20=[x201(:,1) x202(:,1) x203(:,1) x204(:,1)];
%     [~,y20]=max(x20');
%     y20=y20';
% 
%     y201=[y201 y202 y203 y204];
%     y2011=sum(y201')';le1=find(y2011<7);le2=find(y2011==7);le3=find(y2011>7);
%     yy201=[yy201 yy202 yy203 yy204];
%     yy2011=sum(yy201')';lr1=find(yy2011<7);lr2=find(yy2011==7);lr3=find(yy2011>7);

    [confusiono_pre1,confusion_pre1,accuracy_pre1,Precision_pre1,Recall_pre1,Specificity_pre1,F1SCORE_pre1,Kappar_pre1,mcc_pre1]=resultstatsic(y20,test_yyc,size(test_yyc,2),k)
%    
%     block=block+1;
%     confusiono{k,block}=confusiono_pre1;
%     confusion{k,block}=confusion_pre1;
%     acc{k,block}=accuracy_pre1;
%     f1{k,block}=F1SCORE_pre1;
%     kappa(k,block)=Kappar_pre1;
%     mf1(k,block)=mean(F1SCORE_pre1);
%     ac(k,block)=length(find(y20==test_yy1))/length(y20);
%     recall{k,block}=Recall_pre1;
%     specificity{k,block}=Specificity_pre1;
%     rsubs=plottimes(train_yy1,train_y0,yy20,xx20,leng,ss,[1,0],0,0);
%     subs=plottimes(test_yy1,test_y1,y20,x20,leng,ss1,[1,0],0,0);
%     subjectresultr(k,block).fbls=rsubs;
%     subjectresultf(k,block).fbls=subs;
%     predictlab=[predictlab y20];

% rng(256);
% N1=[1 2 8];
% par1=[];
% [RMSE,MAE,traintime,testtime,x1,y1,par1,~,~,~]=AHI_regression(train_x1,train_y,test_x1,test_y,s,C,h,fb1,fb01,par1,pare,N1,N2,fbe1,fe1,nfp,leng1,1);
% 
% rng(256);
% par1=[];
% [RMSE,MAE,traintime,testtime,x2,y2,par1,~,~,~]=AHI_regression(train_xc,train_y,test_xc,test_y,s,C,h,fbloc,fb0loc,par1,pare,N1,N2,fbeloc,fe1,nfp,leng1,1);
% x1=(x1*1.5+x2*0.5)/2;
% y1=(y1*1.5+y2*0.5)/2;

    rng(256);
    par1=[];
%     N1=[1 2 8];
    [RMSE,MAE,traintime,testtime,x,y,par10,~,~,~]=AHI_regression(train_x,train_y,test_x,test_y,s,C,h,fb,fb0,par1,pare,N1,N2,fbe,fe1,nfp,leng1,1);
% x=(x*1.5+x1*0.5)/2;
% y=(y*1.5+y1*0.5)/2;

    rng(256);
    par1=[];
%     N1=[1 2 8];
    [RMSE,MAE,traintime,testtime,x,y,par10,~,~,~]=AHI_regression([train_x x*0.3],train_y,[test_x y*0.3],test_y,s,C,h,fb,fb0,par1,pare,N1,N2,fbe,fe1,nfp,leng1,1);    

%     rng(256)
%     par1=[];
%     [RMSE,MAE,traintime,testtime,x,y,par1,~,~,~]=AHI_regression([train_x train_xc/1.5],train_y,[test_x test_xc/1.5],test_y,s,C,h,fb,fb0,par1,pare,N1,N2,fbe,fe1,nfp,leng1,1);

%     rng(256)
%     par1=[];
%     [RMSE,MAE,traintime,testtime,x1,y1,par1,~,~,~]=AHI_regression([ train_xc],train_y,[test_xc],test_y,s,C,h,fb,fb0,par1,pare,N1,N2,fbe,fe1,nfp,leng1,1);
%     x=(x*1.5+x1*0.5)/2;
%     y=(y*1.5+y1*0.5)/2;
%     rng(256)
%     par1=[];pare=[];
%     fb1=[1 4 1 ];fb01=[1];fbe1={};
%     [RMSE,MAE,traintime,testtime,x,y,par1,~,~,~]=AHI_regression( [x1 xx20 (bmi(marktr)>=24)/1.8 age(marktr)/80 ],train_y,[y1 x20 (bmi(markte)>=24)/1.8 age(markte)/80],test_y,s,C,h,fb1,fb01,par1,pare,N1,N2,fbe1,fe1,nfp,leng1,1);
%     %分多组回归
%     [RMSE,MAE,traintime,testtime,x,y,par1,~,~,~]=AHI_regression(train_xc,train_y,test_xc,test_y,s,C,h,fbloc,fb0loc,par1,pare,N1,N2,fbeloc,fe1,nfp,leng1,1);
%     par1=[];rng(256)
%     [RMSE,MAE,traintime,testtime,x1,y1,par1,~,~,~]=AHI_regression(train_x(lr1,:),train_y(lr1,:),test_x(le1,:),test_y(le1,:),s,C,h,fb,fb0,par1,pare,N1,N2,fbe,fe1,nfp,leng1,1);
%     par1=[];rng(256)
%     [RMSE,MAE,traintime,testtime,x2,y2,par1,~,~,~]=AHI_regression(train_x(lr2,:),train_y(lr2,:),test_x(le2,:),test_y(le2,:),s,C,h,fb,fb0,par1,pare,N1,N2,fbe,fe1,nfp,leng1,1);
%     par1=[];rng(256)
%     [RMSE,MAE,traintime,testtime,x3,y3,par1,~,~,~]=AHI_regression(train_x(lr3,:),train_y(lr3,:),test_x(le3,:),test_y(le3,:),s,C,h,fb,fb0,par1,pare,N1,N2,fbe,fe1,nfp,leng1,1);
%     y=[[y1 le1];[y2 le2];[y3 le3]];
%     y=sortrows(y,2);
%     x=[[x1 lr1];[x2 lr2];[x3 lr3]];
%     x=sortrows(x,2);

%  p=[1.05 1.3 1.2 1.05 1 1 1 1 1];
%     p1=ones(size(xx20,1),9);
%     p2=ones(size(x20,1),9);
%     xloc=round(trans1(x)/10);
%     yloc=round(trans1(y)/10);
%     xloc=xloc+1;xloc(xloc>9)=9;
%     yloc=yloc+1;yloc(yloc>9)=9;
%     for i=1:length(xx20)  
%         p1(i,xloc(i))=1.2;
%     end
%     for i=1:length(x20)
%         p2(i,yloc(i))=1.2;
%     end
%     xx20=xx20.*p.*p1;
%     x20=x20.*p.*p2;
    rng(256)
    par1=[];pare=[];
    fb1=[ 1 size(x20,2) ];fb01=[1 ];fbe1={};
    [RMSE,MAE,traintime,testtime,x,y,par1,~,~,~]=AHI_regression( [  x xx20  ],train_y,[ y  x20 ],test_y,s,C,h,fb1,fb01,par1,pare,N1,N2,fbe1,fe1,nfp,leng1,1);

%     [~,~,statisticosa]=osasta(y,test_y);
%     acc(k,block)=statisticosa.acc;
%     recall{k,block}=statisticosa.recall;
%     specificity{k,block}=statisticosa.specificity;
%     confusion{k,block}=statisticosa.confusion;
%     f1{k,block}=statisticosa.f1;
%     mf1(k,block)=statisticosa.mf1;
%     train_time(k,block)=traintime;
%     test_time(k,block)=testtime;
%     model(k,kk)=par1;

    predictosa(k,block).pretrainosa=x;
    predictosa(k,block).pretestosa=y;
    predictosa(k,block).MAE=MAE;
    predictosa(k,block).RMSE=RMSE;

    
%     rng(256)
%     [RMSE,MAE,traintime,testtime,x,y,par1,con0,train_y,tn]=AHI_regression(train_x1,train_y,test_x1,test_y,s,C,h,fb1,fb01,par2,pare,N1,N2,fbe1,fe1,nfp,leng,1);

end

S{k}=ss1';
S1{k}=ss';

end


isfigure=0;
kk=block
osachat=osacha(ss1);
aget=age(ss1);
sleepdurt=sleepdur(ss1);

mark=[0 5 15 30];
K=[];
for i=1:length(leng)
    na=dataset{i};
    K(i)=sum(leng(1:i));
end
K=[0 K];


s=S{1};
y=predictosa(1,kk).pretestosa;
if normark==1
    yk=find(y>0);
osachat1=trans1(osachat(yk));
y=trans1(y(yk));
elseif normark==2
    tt=t(ss1);
    osachat1=trans1(osachat)./tt;
    y=trans1(y)./tt;
else
    osachat1=osachat;
end
[cy,cosacha,statisticosa]=osasta(y,osachat1,[0 5 15 30]);
statisticosa
[icc_score_total, ~, ~, ~, ~, ~, ~] = ICC([y osachat1], '1-1')

y1=[];
for mj=1:length(leng)
    dataset{mj}
    se=K(mj)+1:K(mj+1);
    if mj==2
       se=se(shhs2loc);
    end
y=[];s=[];
for i=1:k
    s=[s;S{i}(se)];
    y=[y;predictosa(i,kk).pretestosa(se)];
end
    age1=aget(se);
    sleepdur1=sleepdurt(se);
    osacha1=osachat(se);

y=[y s];
y=sortrows(y,2);
y=y(:,1:end-1);
if normark==1
osacha1=trans1(osacha1);
y=trans1(y);
elseif normark==2
    osacha1=trans1(osacha1)./tt(se);
    y=trans1(y)./tt(se);
%     osacha1=osacha1./tt(se);
%     y=y./tt(se);
end
y(y>150)=max(osacha1);
% osacha1=osacha1*60/7./sleepdur1;
% y=y*60/7./sleepdur1;
y1=[y1;y];
% [y1,ps]=mapminmax(y',0,max(osacha));
% mark=mapminmax('apply',[0 5 15 30],ps);
%%  icc up to 87.54 with mf1 down to 65.0 while did not using the mapminmax

%% 使用回归系数回归分类阈值结果和之前几乎没有改变，不值得专门思考
%   and with the mapminmax the mf1 can reach 66.6
% osacha=mapminmax('reverse',osacha',ps);
% osacha=osacha';
% y=mapminmax('reverse',y',ps);
% y=y';

MAE_total=mean(abs(osacha1-y));
RMSE_total=sqrt(abs(mean((osacha1-y).^2)));

[cy,cosacha,statisticosa_total]=osasta(y,osacha1,mark);
[icc_score_total, ~, ~, F, ~, ~, p] = ICC([y osacha1], '1-1');
icc_total=icccal(y,osacha1);
[~,~,statisticosa_total1]=osasta1(y,osacha1,mark);
Corplot(osacha1,y,isfigure)% correlation
BAplot(osacha1,y,isfigure)% Bland-Altman


result(mj).name=dataset{mj};
result(mj).icc_total=icc_score_total;
result(mj).icc_total1=icc_total;
result(mj).acc_total=statisticosa_total.acc;
result(mj).mf1_total=statisticosa_total.mf1;
result(mj).kappa_total=statisticosa_total.kappa;
result(mj).R2_total=statisticosa_total.R;
result(mj).recall_total=statisticosa_total.recall;
result(mj).speficity_total=statisticosa_total.specificity;
result(mj).confusion_total=statisticosa_total.confusion;
result(mj).MAE_total=MAE_total;
result(mj).RMSE_total=RMSE_total;
result(mj).predict_total=y;
result(mj).label_total=osacha1;

result(mj).acc5=statisticosa_total1(1).acc;
result(mj).mf15=statisticosa_total1(1).mf1;
result(mj).kappa5=statisticosa_total1(1).kappa;
% result(mj).R25=statisticosa_total1(1).R;
result(mj).recall5=statisticosa_total1(1).recall(1);
result(mj).speficity5=statisticosa_total1(1).specificity(1);

result(mj).acc15=statisticosa_total1(2).acc;
result(mj).mf115=statisticosa_total1(2).mf1;
result(mj).kappa15=statisticosa_total1(2).kappa;
% result(mj).R215=statisticosa_total1(2).R;
result(mj).recall15=statisticosa_total1(2).recall(1);
result(mj).speficity15=statisticosa_total1(2).specificity(1);

result(mj).acc30=statisticosa_total1(3).acc;
result(mj).mf130=statisticosa_total1(3).mf1;
result(mj).kappa30=statisticosa_total1(3).kappa;
% result(mj).R230=statisticosa_total1(3).R;
result(mj).recall30=statisticosa_total1(3).recall(1);
result(mj).speficity30=statisticosa_total1(3).specificity(1);

% a=intersect(find(age1>=18),find(sleepdur1>=240));
% osacha1=osacha1(a);
% y=y(a);
% MAE_selected=mean(abs(osacha1-y));
% RMSE_selected=sqrt(abs(mean((osacha1-y).^2)));
% [cy,cosacha,statisticosa_selected]=osasta(y,osacha1,mark);
% [icc_score_selected, ~, ~, F, ~, ~, p] = ICC([y osacha1], '1-1');
% icc_selected=icccal(y,osacha1);
% if isfigure
% figure
% scatter(osacha1,y,'*')
% hold on
% plot(0:100,0:100)
% end
% 
% result(mj).icc_selected=icc_score_selected;
% result(mj).icc_selected1=icc_selected;
% result(mj).acc_selected=statisticosa_selected.acc;
% result(mj).mf1_selected=statisticosa_selected.mf1;
% result(mj).recall_selected=statisticosa_selected.recall;
% result(mj).speficity_selected=statisticosa_selected.specificity;
% result(mj).confusion_selected=statisticosa_selected.confusion;
% result(mj).MAE_selected=MAE_selected;
% result(mj).RMSE_selected=RMSE_selected;
end
toc
savename=[outdir,'AHIprediction_nodemographic.mat'];%meth为使用方法类型，可在程序中设置
save(savename,"result","Mark");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [breathfeatures,breathfeatures1,demofeatures,f0,label,label1,sleepeff,sleepdur,age,bmi,gender,issmoke,fb,fb0,fbe,fb1,fb01,fbe1,fbloc,fb0loc,fbeloc,leng,p1,p2,mark1,M]=datacollection(dataset,Dir)
breathfeatures=[];label=[];sleepeff=[];age=[];bmi=[];sleepdur=[];gender=[];issmoke=[];f0=[];breathfeatures1=[];label1=[];
fbloc=[];fb0loc=[];fbeloc=[];
load('spo2ahi.mat')
for i=1:length(dataset)
    add=dataset{i};
    filename=[Dir,'\',add,'_spo2feature_total.mat'];
    load(filename)
    load('demography_selected.mat',add)
    eval(['osacha=',add,'(:,8);']);%ahi4 9   a0h3a  8   a0ah3  6  6的效果最好
    eval(['osacha1=',add,'(:,7);']);%ahi4 9   a0h3a  8   a0ah3  6  6的效果最好
    eval(['sleepdur1=',add,'(:,18);']);%sleepdur
    eval(['sleepeff1=',add,'(:,19);']);%sleepdur
    eval(['bmi1=',add,'(:,5);']);
    eval(['age1=',add,'(:,3);']);
    eval(['gender1=',add,'(:,2);']);
    eval(['issmoke1=',add,'(:,20);']);
    if strcmp(add,'mesa')
       issmoke1=zeros(length(age1),1);
    end
%     if strcmp(add,'shhs1')
%         m=setdiff(1:size(shhs1,1)-3,[2066 2072 2843]);
%         shhs1=shhs1(m,:);
%         spo2feature=spo2feature(m,:);
%         osacha=osacha(m,:);
%     end
%% napai

mark=find(sleepdur1<240);
if  i==1 || i==2 || i==3
os=osacha-osacha1;
osachac=ahiclass(osacha);
osacha1c=ahiclass(osacha1);
os1=abs(osachac-osacha1c);
mark=union(mark,union(intersect(find(abs(os)>=10),find(os1>=1)),spo2ahi(i).diff));
end
if i==1
   mark1=intersect(mark,shhs1loc);
end
   mark=setdiff(1:length(sleepeff1),mark);
if i==1
   mark1=union(mark1,mark(end-562:end));
   [~,ia]=intersect(shhs1loc,mark);
   ls1=length(mark);
end
if i==2
   [~,mark1]=intersect(mark,ia);
   mark1=mark1([20:120 562:end]);
end
if i==1
    rng(33)
    markt=randperm(length(mark));
    M{i}=markt(end-561:end);
elseif i==2
    M{i}=mark1;
else
    M{i}=mark;
end
% if i==3
%    mark=union(mark,find(osacha>30));
% end
%%
%     if strcmp(add,'cfs')
%     mark=setdiff(mark,34);
%     end
%%% local feature
% [~,f,fbloc,fb0loc,fbeloc,~,~]=breath3600s(add);
% if strcmp(add,'shhs1')
%    f=[f(1:2065,:);f(2065,:);f(2066:2071,:);f(2071,:);f(2072:2842,:);f(2842,:);f(2843:end,:) ];
% end
% %%%
%     f0=[f0;f(mark,:)];
    breathfeatures=[breathfeatures;spo2feature_nowake(mark,:)]; % spo2feature_nowake
    breathfeatures1=[breathfeatures1;spo2feature_nowake1(mark,:)];
    label=[label;osacha(mark,:)];
    label1=[label1;osacha1(mark,:)];
    sleepdur=[sleepdur;sleepdur1(mark,:)];
    sleepeff=[sleepeff;sleepeff1(mark,:)];
    age=[age;age1(mark,:)];bmi=[bmi;bmi1(mark,:)];
    gender=[gender;gender1(mark,:)];
    issmoke=[issmoke;issmoke1(mark,:)];
    leng(i)=length(mark);
end

mark=1:size(breathfeatures,2);
breathfeatures=breathfeatures(:,mark);
demofeatures=[age/100 sleepeff/100 bmi sleepdur gender/2 issmoke/2.5];
fb=[4 1 1 1 1   1 1 1 1 1 1   1 2 2   1 1 1 1 ];  fb1=[  1 2 2 2 2  2 2 2 1 2 1 1];
fb0=[[1 2],[6 7],[9:14],[18 19],[20:23]]; fb01=[[24:26],[33:34]];
fbe={[1 2],[6 7],[9:14],[18 19],[20:23]}; fbe1={[24:26]-23,[33:34]-23};
p1=1:sum(fb);p2=sum(fb)+1:sum(fb)+sum(fb1);

breathfeatures=delout(breathfeatures);

breathfeatures(:,[8  ])=breathfeatures(:,[8  ])*100;%mapminmax(breathfeatures(:,[8 17 23 32:33 43])')';
breathfeatures(:,[17 23])=breathfeatures(:,[17 23])*100;
breathfeatures(:,[24:26 37 38])=breathfeatures(:,[24:26 37 38])./100;
breathfeatures(:,[35 36 ])=breathfeatures(:,[35 36 ])/10;

end

function Corplot(osacha1,y,isfigure)
if isfigure
figure
scatter(osacha1,y,'*')
hold on
plot(0:max(y),0:max(y))
end
end

function BAplot(osa,y,isfigure)
if isfigure==1
figure
y(y<0)=0.1;
dif=osa-y;
[osa1,o1]=sort(osa);
scatter((osa1+y(o1))/2,dif(o1),'filled')
r=round(max((osa1+y(o1))/2));
hold on
plot(mean(dif)*ones(1,r),'LineStyle','--','Color','Yellow')
plot(1.96*std(dif)*ones(1,r),'LineWidth',2,'Color','red')
plot(-1.96*std(dif)*ones(1,r),'LineWidth',2,'Color','red')
hold off
end
end

function a1=ahiclass(a)
n=[0 5 15 30 2000];
a1=ones(size(a,1),1);
for i=2:length(n)-1
    n1=n(i);n2=n(i+1);
    a1(intersect(find(a>=n1),find(a<n2)))=i;
end
end

function dat=delout(dat)
md=mean(dat);
sd=std(dat);
mad=max(dat);
st1=md+3*sd;
s=mad>st1;
for i=1:length(s)
    if s(i)==1
        dat(find(dat(:,i)>st1(i)*1.2),i)=st1(i);
    end
end
end

function [f,f0,fb,fb0,fbe,labelb,lengosa]=breath3600s(ad)
Dir=['D:\E\sleep\open database\breathfeatures\features'];
Dir2=['D:\E\sleep\open database\features'];
filename3=[Dir,'\',ad,'_spo2features300s.mat'];
filename2=[Dir2,'\',ad,'\',ad,'.mat'];
filename5=[Dir,'\',ad,'_breath_label_ahi.mat'];
load(filename3)
load(filename2,'label','leng')
load(filename5)
u=10;

sleepmark=[];
labelmark=[];
for i=1:length(len300)
lab=zeros(len300(i),1);
    for j=1:ceil(leng(i)/u)
    lab1=label((j-1)*u+1:min(j*u,leng(i)));
    if sum(lab1)>7
        lab(j)=1;
    end
    end
    labelmark=[labelmark;lab];
end
if isempty(sleepmark)
   sleepmark=1:length(spo2features300s);
end

labelb=Labelo300;%Labelo1 Labelo30 Labelb1 Labelb30
labelb(labelb==0)=2;%如果用事件Labelb，这里用3

for i=1:length(len300)
    k(i)=sum(len300(1:i));    
end
fma=[];
for i=1:length(len300)
fma=[fma;(1:len300(i))'/len300(i)*0.7];
end
%% sleeples 样本筛选
fma=fma(sleepmark);
spo2features=spo2features300s(sleepmark,:);
labelb=labelb(sleepmark);
spo2features(:,14)=spo2features(:,14)/1.5;

f=[spo2features(:,[ 1 2 3:6  7 8    11:13   21 23 25 27     ] )];% spo2dis
fspo2=[ 1 1 4 2   3    4  ];%  1 4 2 3   1 2 1 1 1   2 2 2 2 4  1 4 1 1 1
% f=[spo2features(:,[ 1 2 3:6  7 8    11:13 14  15 16  21 23 25 27     ] )];% spo2dis
% fspo2=[ 1 1 4 2    3    1 2 4  ];%  1 4 2 3   1 2 1 1 1   2 2 2 2 4  1 4 1 1 1
fb=[fspo2];
fb0=[ 1 2 [3 5 6 11 12 13]  ];
fbe={  1;2; [3 5:6] ; [11 12 13]};
f(:,[4 5 6])=f(:,[4 5 6])*1.5;

mark=spo2features(:,end);%最后一行为标志位，0为可用
mark1=sum(isinf(f)')';
mark2=sum(isnan(f)')';
mark=mark+mark1+mark2;
mark(setdiff(1:length(mark),sleepmark))=1;
k=[];
if sum(leng)>size(spo2features,1)%%%%%%%
leng=len300;%%%%%
end%%%%%%
for i=1:length(leng)
    k(i)=sum(leng(1:i));    
end
k=[0 k];

for i=1:length(leng)
    lengosa(i)=length( find(mark(k(i)+1:k(i+1))==0) );
end
sleeples=find(lengosa~=0);
lengosa=lengosa(sleeples);

mark=find(mark==0);
mark=intersect(mark,sleepmark);
labelb=labelb(mark,:);
f=[f(mark,:)];%加入wakelabel

%%截取信号长度
n=7;
for i=1:length(lengosa)
     K(i)=sum(lengosa(1:i));
end
K=[0 K];
for i=1:length(lengosa)
    f1=f(K(i)+1:K(i+1),:);
    if size(f1,1)>n*12;
        f1=f1(1:n*12,:);
    else
        f1=[f1;zeros(n*12-size(f1,1),size(f1,2))];
    end

    f2=[];
    for j=1:n
        f2=[f2 median(f1((j-1)*12+1:min(j*12,size(f1,1)),:))];
    end
%     if j>=n
%         f2=[f2(1:n*size(f1,2)) ];
%     else
%         f2=[f2  repmat(zeros(1,size(f1,2)),1,n-round(lengosa(i)/12)) ];
%     end
    f2=[f2 mean(f1)];
    f0(i,:)=f2;
end
fb01=fb0;fbe1=fbe;
fb=repmat(fb,1,n+1);
for i=1:n
fb0=[fb0 fb01+i*size(f1,2)];
for j=1:4
    fbe2{j}=fbe1{j}+i*size(f1,2);
end
fbe=[fbe;fbe2'];
end
end


function icc=icccal(y,osachat1)
% nc 中的计算方法
a1=mean(y.^2);
a2=mean(abs(osachat1-y).^2);
a3=mean(osachat1.^2);
icc=(a1-a2)/(a1+a2+2*(a2-a2)/length(y));
end
% function [ x, y ,cx, cy, MAE, RMSE] = osares(leng,ss,ss1,yy20,y20,train_xb,test_xb,osacha,bmi,sleepeff,label,age)
% [train_x,train_y,fo,f0,fe]=geosada(leng,ss,yy20,train_xb,osacha,bmi,sleepeff,label,age);
% [test_x,test_y,~,~,~]=geosada(leng,ss1,y20,test_xb,osacha,bmi,sleepeff,label,age);
% par1=[];pare=[];
% rng(256)
% N1=[1 2];
% N2=1;
% s=0.5;C=0.02;
% h=1;fe1=[];
% nfp=0;leng=ones(1,length(leng));
% [RMSE,MAE,traintime,testtime,x,y,par1,con0,train_y,tn]=AHI_regression(train_x,train_y,test_x,test_y,s,C,h,fo,f0,par1,pare,N1,N2,fe,fe1,nfp,leng,1);
% % x=x*10;
% % y=y*10;
% % 把y变换，靠近test_y
% [~,~,statisticosa]=osasta(y,test_y,mark)
% statisticosa.mf1
% 
% [cy,ly]=osalabel(y);
% [cx,lx]=osalabel(x);
% end

function [y,ly]=osalabel(osacha)
y=double([osacha<=5 (osacha<=15 & osacha>5) (osacha<=30 & osacha>15) osacha>30]);
[~,ly]=max(y');
ly=ly';
end

function da1=extime(da,t)
ti=t/30;
for i=1:ceil(length(da)/ti)
    da0=da((i-1)*ti+1:min(i*ti,length(da)),:);

    da1(i)=rem(sum(da0),2);
end
end

function [osadatr,osatr,fo,f0,fe]=geosada(leng,ss,yy20,x,osacha,bmi,sleepeff,label,age)
k=0;lengss=leng(ss);lenm=mean(leng);
for i=1:length(ss)
    k(i)=sum(lengss(1:i));
end
k=[0 k];
for i=1:length(ss)
    la=label(k(i)+1:k(i+1));
    lac=find(la==1);
    lac1(i,1)=length(lac)*5/60/10;

    xa=x(k(i)+1:k(i+1),:);
    da=yy20(k(i)+1:k(i+1),:);


    [~,da1]=max(da');
    da1=da1';

    osadatr(i,1)=length(find(da(:,1)>0.5))/length(da);
    osadatr(i,2)=length(find(da(:,2)>0.5))/length(da);
    osadatr(i,3)=length(find(diff(da1)~=0))/length(da1);
    osadatr(i,4)=length(find(diff(da1)==0))/length(da1);
% osadatr(i,5)=length(find(xa(:,3)+xa(:,4)<0.95))/length(xa);
    osadatr(i,5)=length(find(xa(:,3)<0.9))/length(xa);
    osadatr(i,6)=length(find(xa(:,5)>0.1))/length(xa);
    osadatr(i,7)=length(find(xa(:,6)>=0.02))/length(xa);

    osadatr(i,8)=sqrt(mean(xa(:,11)));
    osadatr(i,9)=sqrt(mean(xa(:,12)));

    % osadatr(i,10)=mean(xa(:,15));
    % osadatr(i,11)=mean(xa(:,16));
    osadatr(i,10)=length(find(xa(:,15)<0.4))/length(xa);
    osadatr(i,11)=length(find(xa(:,15)>0.3))/length(xa);

end
if mean(sleepeff)<1
    sleepeff=sleepeff*100;
end
con1=bmi(ss)>24; 
con2=sleepeff(ss)/100;
con3=age(ss)/1000;
con=con2-con1*0.1-con3;
osadatr=[osadatr  con  ];%看lac1是否有别的用法  lac1

fo=[2 2 3 1 1 1  1];
f0=[1 2];
fe={[1 2];[3 4];[5 6];8;size(osadatr,2)};

% y=double([osacha<=5 (osacha<=15 & osacha>5) (osacha<=30 & osacha>15) osacha>30]);
osatr=osacha(ss);
end

function [a1,b1,statistic]=osasta(a,b,mark)
n=[mark 2000];
m=[mark 2000];
a(a<0)=0.1;
a1=ones(size(a,1),1);
b1=ones(size(b,1),1);
for i=2:length(n)-1
    n1=n(i);n2=n(i+1);
    m1=m(i);m2=m(i+1);
    a1(intersect(find(a>=n1),find(a<n2)))=i;
    b1(intersect(find(b>=m1),find(b<m2)))=i;
end
    ac=length(find(a1==b1))/length(a1);
    [con,Acc_part_1,accuracy_1,precision_1,recall_1,specificity_1,F1score_1,kappa_1,mcc_1] = Evaluation(a1,label2lab(b1),length(mark));
    statistic.ahi=n;
    statistic.acc=ac;
    statistic.recall=recall_1;
    statistic.specificity=specificity_1;
    statistic.confusion=con;
    statistic.f1=F1score_1;
    statistic.mf1=2*mean(precision_1)*mean(recall_1)/(mean(precision_1)+mean(recall_1));
    statistic.kappa=kappa_1;
    statistic.R=1-sum((b-a).^2)/sum((b-mean(a)).^2);
end

function [a1,b1,statistic]=osasta1(a,b,mark)
n=[mark];
m=[mark];
a(a<0)=0.1;

for i=1:length(n)-1
    a1=ones(size(a,1),1);
    b1=ones(size(b,1),1);
    n1=n(i+1);
    m1=m(i+1);
    a1(a>=n1)=2;
    b1(b>=m1)=2;
    ac=length(find(a1==b1))/length(a1);
    [con,Acc_part_1,accuracy_1,precision_1,recall_1,specificity_1,F1score_1,kappa_1,mcc_1] = Evaluation(a1,label2lab(b1),2);
    statistic(i).theroshod=n1;
    statistic(i).acc=ac;
    statistic(i).recall=recall_1;
    statistic(i).specificity=specificity_1;
    statistic(i).f1=F1score_1;
    statistic(i).mf1=2*mean(precision_1)*mean(recall_1)/(mean(precision_1)+mean(recall_1));
    statistic(i).kappa=kappa_1;
    statistic(i).R=1-sum((b-a).^2)/sum((b-mean(a)).^2);
end
end

function [group1,group2,group3,group4,groupc,indictg1,indictg2,indictg3,indictg4,indictgc]=groupdat(age,osa,folds)
[~,b]=hist(age,4);o=[0 5 15 30];
for i=1:length(age)
    if age(i)<b(2)
        sub(i,1)=1;
    elseif age(i)<b(3) && age(i)>=b(2)
        sub(i,1)=2;
    elseif age(i)<b(4) && age(i)>=b(3)
        sub(i,1)=3;
    elseif age(i)>b(4)
        sub(i,1)=4;
    end
    if osa(i)<o(2)
        sub(i,2)=1;
    elseif osa(i)<o(3) && osa(i)>=o(2)
        sub(i,2)=2;
    elseif osa(i)<o(4) && osa(i)>=o(3)
        sub(i,2)=3;
    elseif osa(i)>o(4)
        sub(i,2)=4;
    end
end

group1=intersect( find(sub(:,1)==1),find(sub(:,2)==1) );
group2=intersect( find(sub(:,1)==2),find(sub(:,2)==2) );
group3=intersect( find(sub(:,1)==3),find(sub(:,2)==3) );
group4=intersect( find(sub(:,1)==4),find(sub(:,2)==4) );
groupg=[group1;group2;group3;group4];
groupc=setdiff(1:length(osa),groupg);

indictg1=geindict(length(group1),folds);
indictg2=geindict(length(group2),folds);
indictg3=geindict(length(group3),folds);
indictg4=geindict(length(group4),folds);
indictgc=geindict(length(groupc),folds);

end

function indict=geindict(hd,flods)
indict=ones(1,hd);
remk=rem(1:hd,flods);
for i=1:flods
    lk=find(remk==i-1);
    indict(lk)=i;
end
indict=indict';
end

function [ss,ss1,mixsub]=gettrte(group1,group2,group3,group4,groupc,indictg1,indictg2,indictg3,indictg4,indictgc,k,len)
i1=group1(find(indictg1==k));
i2=group2(find(indictg2==k));
i3=group3(find(indictg3==k));
i4=group4(find(indictg4==k));
i5=groupc(find(indictgc==k));
mixsub=setdiff(groupc,i5);
ss1=[i1;i2;i3;i4;i5'];
ss=setdiff(1:len,ss1);
end

function [leng,sleeples,sleepmark,osacha,bmi,age,sleepdur,sleepeff]=delsleepless(leng,sleepdur,osacha,bmi,age,sleepeff,n,unit)
sleeples=sleepdur/60>n;%%挑选睡眠较少的  4

k=[];
u=unit/30;
leng=ceil(leng/u);
for i=1:length(leng)
    k(i)=sum(leng(1:i));
end
k=[0 k];
sleepmark=zeros(sum(leng),1);
for i=1:length(leng)
    sleepmark(k(i)+1:k(i+1))=sleeples(i);
end
sleepmark=find(sleepmark==1);

leng=leng(sleeples);
osacha=osacha(sleeples);
bmi=bmi(sleeples);
sleepdur=sleepdur(sleeples);
sleepeff=sleepeff(sleeples);
age=age(sleeples);
end

function [bremark,leng,label1,sleepdur]=breathsample(leng,label,sleeples,unit,len)

k=[];
for i=1:length(leng)
    k(i)=sum(leng(1:i));
end
k=[0 k];
labelmark=zeros(sum(leng),1);
for i=1:length(leng)
    labelmark(k(i)+1:k(i+1))=sleeples(i);
end
labelmark=find(labelmark==1);
label=label(labelmark);
leng=leng(sleeples);
k=[];
for i=1:length(leng)
    k(i)=sum(leng(1:i));
end
k=[0 k];

k1=[];
u=unit/30;
leng1=ceil(leng/u);
for i=1:length(leng)
    k1(i)=sum(leng1(1:i));
end
k1=[0 k1];
label1=[];
bremark=[];
sleepdur=[];
for i=1:length(leng)
    lab=label(k(i)+1:k(i+1));

    for j=1:floor(length(lab)/u)
        lab11=lab(u*(j-1)+1:u*j);
        if length(find(lab11==0))>=3
            lab1(j)=0;
        else
            lab1(j)=1;
        end
    end
%     a=find(lab1~=0,1,"first");
%     b=find(lab1~=0,1,"last");
%     a=a:b;
    a=find(lab1~=0);
    
%     ch=len-length(a);% 将信号补齐为len
%     if ch<=0
%        a=a(1:ch);
%     else
%        a=[a a(1:ch)];
%     end
    sleepdur=[sleepdur length(a)];
    m=(a)+k1(i);

    bremark=[bremark m];
    leng(i)=length(m);
    label1=[label1;lab(a)];
end
% bremark=intersect(bremark,sleepmark);
end

function [f,fcon,fb,fb0,fbe,labelb]=norbreattf_1s(Dir)
filename3=[Dir,'_spo2features1s.mat'];
filename5=[Dir,'_breath_label.mat'];
load(filename3)
load(filename5)
len=length(Labelo30);
labelb=Labelo1;%Labelo1 Labelo30 Labelb1 Labelb30  o osa仅区分osa事件  b区分低通气和呼吸暂停
labelb(labelb==0)=2;
fcon=SpO2con1s;
f=spo2features1s(:,[   4 9 14  5 10 15  6 11 16  7 12 17]);%1 2     3 8 13
fb=[3*ones(1,4)];%[1 1 5 5 5]; 1 1
fb0=[  7 8  13 14]-5;%1 2    
fbe={[6 7 8]-2;[12 13 14]-5};%[1 2];    
end

function [f,fb,fb0,fbe,labelb,lengosa,osacha,bmi,age,sleepeff,sleepdur,labelmark]=norbreattf300s(Dir,leng,sleepmark,osacha,bmi,age,sleepeff,sleepdur,label)
filename3=[Dir,'_spo2features300s.mat'];
filename5=[Dir,'_breath_label_ahi.mat'];
load(filename3)
load(filename5)
u=10;

labelmark=[];
for i=1:length(len300)
lab=zeros(len300(i),1);
    for j=1:ceil(leng(i)/u)
    lab1=label((j-1)*u+1:min(j*u,leng(i)));
    if sum(lab1)>7
        lab(j)=1;
    end
    end
    labelmark=[labelmark;lab];
end
if isempty(sleepmark)
   sleepmark=1:length(spo2features300s);
end

labelb=Labelo300;%Labelo1 Labelo30 Labelb1 Labelb30
labelb(labelb==0)=2;%如果用事件Labelb，这里用3

for i=1:length(len300)
    k(i)=sum(len300(1:i));    
end
fma=[];
for i=1:length(len300)
fma=[fma;(1:len300(i))'/len300(i)*0.7];
end
%% sleeples 样本筛选
fma=fma(sleepmark);
spo2features=spo2features300s(sleepmark,:);
labelb=labelb(sleepmark);
% spo2features(:,9)=mapminmax(spo2features(:,9),0,0.5);
% spo2features(:,10)=mapminmax(spo2features(:,10),0,0.5);
spo2features(:,14)=spo2features(:,14)/1.5;
% spo2features(:,15)=sum(spo2features(:,15:17)')';

% [ones(1,14) 3 1 1 1 2 2 2 2]
%%% spo2 30+15
f=[spo2features(:,[ 1 2 3:6  7 8    11:13 14  15 16  21 23 25 27     ] )];% spo2dis
% 
% fspo290=[1 4 2];fspo230=[];fspo2diff=[3];
fspo2=[ 1 1 4 2    3    1 2 4  1 1 ];%  1 4 2 3   1 2 1 1 1   2 2 2 2 4  1 4 1 1 1
fb=[fspo2];
fb0=[ 1 2 [3 5 6 11 12 13]  ];
fbe={  1;2; [3 5:6] ; [11 12 13]};
f(:,[4 5 6])=f(:,[4 5 6])*1.5;

% mark=zeros(size(f,1),1);
mark=spo2features(:,end);%最后一行为标志位，0为可用
mark1=sum(isinf(f)')';
mark2=sum(isnan(f)')';
mark=mark+mark1+mark2;
mark(setdiff(1:length(mark),sleepmark))=1;
k=[];
if sum(leng)>size(spo2features,1)%%%%%%%
leng=len300;%%%%%
end%%%%%%
for i=1:length(leng)
    k(i)=sum(leng(1:i));    
end
k=[0 k];

for i=1:length(leng)
    lengosa(i)=length( find(mark(k(i)+1:k(i+1))==0) );
end
sleeples=find(lengosa~=0);
lengosa=lengosa(sleeples);
osacha=osacha(sleeples);
bmi=bmi(sleeples);
age=age(sleeples);
sleepeff=sleepeff(sleeples);
sleepdur=sleepdur(sleeples);


mark=find(mark==0);
mark=intersect(mark,sleepmark);
labelb=labelb(mark,:);
f=[f(mark,:)  fma(mark) labelmark(mark)*0.5];%加入wakelabel
% label1=label(mark);

% f(:,[30 37]-1)=f(:,[30 37]-1)/6;
% f=zscore(f')';
end

function [f,fcon,fb,fb0,fbe,labelb,lengosa,label1]=norbreattfs(Dir,leng,sleepmark,label)
filename3=[Dir,'_spo2features30s+15s.mat'];
filename5=[Dir,'_breath_label_ahi.mat'];
load(filename3)
load(filename5)
% fcon=ones(size(SpO2con,1),1);
fcon=SpO2con;
fcon(fcon~=1)=eps;
% s=find(con==1);
% fcon(s)=2;
tl=3;label1=[];
labelb=Labelo30;%Labelo1 Labelo30 Labelb1 Labelb30
labelb(labelb==0)=2;%如果用事件Labelb，这里用3

spo2features=spo2features(sleepmark,:);
labelb=labelb(sleepmark);
spo2features(:,9)=mapminmax(spo2features(:,9),0,0.5);
spo2features(:,10)=mapminmax(spo2features(:,10),0,0.5);
spo2features(:,15)=sum(spo2features(:,15:18)')';
spo2features(:,16)=sum(spo2features(:,19:21)')';
%%% spo2 30+15
f=[spo2features(:,[ 2 3:6  7 8  9 10  11:13 14:15  22 23  24 26 28 30   ] )];% spo2dis
fspo290=[1 4 2];fspo230=[];fspo2diff=[3];
fspo2=[ 1 4 2  2  3    1 1  1 1   4 ];%  1 4 2 3   1 2 1 1 1   2 2 2 2 4  1 4 1 1 1
fb=[fspo2];
fb0=[ 1 [3 5 6 11 12 13]  ];
fbe={  1; [3 5:6] ; [11 12 13]; };

for i=1:length(leng)
    k(i)=sum(leng(1:i));    
end
mark=spo2features(:,end);%最后一行为标志位，0为可用
mark1=sum(isinf(f)')';
mark2=sum(isnan(f)')';
mark=mark+mark1+mark2;
k=[0 k];
for i=1:length(leng)
    lengosa(i)=length( find(mark(k(i)+1:k(i+1))==0) );
end

mark=find(mark==0);
labelb=labelb(mark,:);
f=f(mark,:);
label1=label(mark);
% f(:,[30 37]-1)=f(:,[30 37]-1)/6;
% f=zscore(f')';
end

function [f,fcon,fb,fb0,fbe,labelb,lengosa]=norbreattf_90s(Dir,leng,sleepmark)
filename3=[Dir,'_spo2features30s.mat'];
filename5=[Dir,'_breath_label.mat'];
load(filename3)
load(filename5)
% fcon=ones(size(SpO2con,1),1);
fcon=SpO2con;
fcon(fcon~=1)=eps;
% s=find(con==1);
% fcon(s)=2;
tl=3;
labelb=Labelo30;%Labelo1 Labelo30 Labelb1 Labelb30
labelb(labelb==0)=2;%如果用事件Labelb，这里用3

spo2features=spo2features(sleepmark,:);
labelb=labelb(sleepmark);

%%% spo2 30
f=[spo2features(:,[1 3:8  11:13 18 19  22 24 26   23 25 27 ] )];% spo2dis
fspo290=[1 4 2];fspo230=[];fspo2diff=[3];
fspo2=[1 4 2 3 2  3 3  ];%  1 4 2 3   1 2 1 1 1   2 2 2 2 4  1 4 1 1 1
fb=[fspo2];
fb0=[ 1 [3 5 6 11 12 13]-1  ];
fbe={  1; [3 5:6]-1 ; [11 12 13]-1; };


% %%% spo2  90
% f=[spo2features90s(:,[1 3:8  11:13 18:19  21 23 25 27 28 30:32 33:end-4 ] )];% spo2dis
% fspo290=[1 4 2];fspo230=[];fspo2diff=[3];
% fspo2=[1 4 2 3 2   4  4   4  ];%  1 4 2 3   1 2 1 1 1   2 2 2 2 4  1 4 1 1 1
% fb=[fspo2];
% fb0=[ 1 [3 5 6 11 12 13]-1  ];
% fbe={  1; [3 5:6]-1 ; [11 12 13]-1; };

%%% icc  0.7776
% f=[spo2features90s(:,[1 3:8  11:13 15:17 20:28 29:32 34:end-3 ] )];% spo2dis
% fspo290=[1 4 2];fspo230=[];fspo2diff=[3];
% fspo2=[1 4 2 3 2 1  2 2 2 2 4  4 1  ];%  1 4 2 3   1 2 1 1 1   2 2 2 2 4  1 4 1 1 1
% fb=[fspo2];
% fb0=[ 1 [3 5 6 11 12 13]-1  ];
% fbe={  1; [3 5:6]-1 ; [11 12 13]-1; };

%icc 0.7676  但 acc高 71.5   mf1  0.68
% f=[spo2features90s(:,[1 3:8  11:13 18:19 20:27 28 30:32 33:end-4 ] )];% spo2dis
% fspo290=[1 4 2];fspo230=[];fspo2diff=[3];
% fspo2=[1 4 2 3 2  2 2 2 2  4   4  ];%  1 4 2 3   1 2 1 1 1   2 2 2 2 4  1 4 1 1 1
% fb=[fspo2];
% fb0=[ 1 [3 5 6 11 12 13]-1  ];
% fbe={  1; [3 5:6]-1 ; [11 12 13]-1; };

for i=1:length(leng)
    k(i)=sum(leng(1:i));    
end
mark=spo2features(:,end);%最后一行为标志位，0为可用
mark1=sum(isinf(f)')';
mark2=spo2features(:,1)==0.5;
mark=mark+mark1+mark2;
k=[0 k];
for i=1:length(leng)
    lengosa(i)=length( find(mark(k(i)+1:k(i+1))==0) );
end

mark=find(mark==0);
labelb=labelb(mark,:);
f=f(mark,:);
% f(:,[30 37]-1)=f(:,[30 37]-1)/6;
f=zscore(f')';
end

function [f,fcon,fb,fb0,fbe,labelb,lengosa]=norbreattf(Dir,leng,sleepmark)
filename3=[Dir,'_spo2features.mat'];
filename4=[Dir,'_thermfeatures.mat'];
filename5=[Dir,'_breath_label.mat'];
load(filename3)
load(filename4)
load(filename5)
% fcon=ones(size(SpO2con,1),1);
fcon=SpO2con;
fcon(fcon~=1)=eps;
% s=find(con==1);
% fcon(s)=2;
tl=3;
labelb=Labelo30;%Labelo1 Labelo30 Labelb1 Labelb30
labelb(labelb==0)=2;%如果用事件Labelb，这里用3


% spo2features=spo2features(sleepmark,:);
% labelb=labelb(sleepmark);
%%% spo2 (46)
spo2dis=neithbourdis(spo2features(:,[1:3]),tl);
f=[spo2features(:,[ 1 2 3:6  9:10 11 14 17 20 12 15 18 21  [23 26 29 32 35 38 ] [23 26 29 32 35 38 ]+1 ] )];% spo2dis
fspo2=[4 1 1 1 1  4 4  4 1 1   4 1 1   ];
fb=[fspo2];
fb0=[ 5 6 [5 6]+15  [5 6]+15+6 ];
fbe={ 1:4 ; [1 3:5]+15 ; [1 3:5]+15+6 };
f(:,3)=1-f(:,3);
f(:,[1 2 5 6 9:18 23 24])=f(:,[1 2 5 6 9:18 23 24])/2;
% f(:,[23 24 31 32])=f(:,[23 24 31 32])/8;
%%%%%
mark=find(f(:,1)~=0.5);
for i=1:length(leng)
    k(i)=sum(leng(1:i));    
end
k=[0 k];
for i=1:length(leng)
    lengosa(i)=length( find(f(k(i)+1:k(i+1),1)~=0.5) );
end
f=f(mark,:);
labelb=labelb(mark);

% %%% new version(spo2 22 )         spo2+therm
% spo2dis=neithbourdis(spo2features(:,[1:3]),tl);
% thermdis=neithbourdis(thermfeatures(:,13),tl);
% f=[spo2features(:,[ 1 3:6  9:10 11 14 17 20 12 15 18 21]) spo2dis thermfeatures thermdis];%*1.5.*fcon;%(:,[1 3 4 5 7 8 9 11 12 13])
% % f=[spo2features(:,1:5)];
% fspo2=[3 1 1 1 1 4 4 (tl*2+1)*ones(1,3)]; 
% ftherm=[4 4 4 1 (tl*2+1)*ones(1)];
% fb=[fspo2 ftherm];
% sfspo2=sum(fspo2);
% fb0=[5 6 sfspo2+9];
% fbe={1:4;sfspo2+14:sfspo2+16;[1 sfspo2+1 sfspo2+4];[2 sfspo2+2 sfspo2+5];[3 sfspo2+3 sfspo2+6]};
% %%%%%

% %%% old version (spo2 8)     spo2+therm
% spo2dis=neithbourdis(spo2features(:,[1:3]),tl);
% thermdis=neithbourdis(thermfeatures(:,13),tl);
% f=[spo2features(:,1:5) spo2dis thermfeatures thermdis]*1.5.*fcon;%(:,[1 3 4 5 7 8 9 11 12 13])
% % f=[spo2features(:,1:5)];
% fspo2=[4 1 (tl*2+1)*ones(1,3)];
% ftherm=[4 4 4 1 (tl*2+1)*ones(1)];
% fb=[fspo2 ftherm];
% fb0=[5 fspo2+9];
% fbe={1:4;28:30;[1 15 19];[2 16 20];[3 17 21]};
% %%%%%
end

function b=neithbourdis(a,n)
a=[zeros(1,size(a,2));a];
a1=time_causal(a,n,2);
b=diff(a1);
end

function [trainacc,testacc,xx1,x1,yy1,y1,par1,con0,train_y,tn]=blsc(train_x,train_y,test_x,test_y,s,C,h,fo,f0,par1,pare,N1,N2,fe,fe1,nfp,lengo,ss1)
    [trainacc,traintime,xx1,yy1,par1,con0,train_y,tn] = multi_scale_bls41(train_x,train_y,test_x,test_y,s,C,h,fo,f0,par1,[],1,N1,N2,fe,fe1,[],nfp,[]);
%     leng0=lengo(ss1);
%     leng0=[];
%     for i=1:length(leng01)
%         leng0=[leng0 leng01{i}];
%     end
if length(ss1)<length(lengo)
        leng0=[];
        for lp=1:length(ss1)
            leng0=[leng0 lengo(ss1(lp))];
        end
    for xi=1:length(leng0)
        K0(xi)=sum(leng0(1:xi));
    end
    K0=[0 K0];
else
K0=[0 lengo];
end
    [testacc,testtime,x1,y1,par1,~,~] = multi_scale_bls41(train_x,train_y,test_x,test_y,s,C,h,fo,f0,par1,[],2,N1,N2,fe,fe1,con0,nfp,K0);
    clear K0
    trainacc=trainacc(end);
end

function [trainacc,testacc,xx1,x1,yy1,y1,par1,con0,train_y,tn]=blsc1(train_x,train_y,test_x,test_y,s,C,h,fo,f0,par1,pare,N1,N2,fe,fe1,nfp,lengo,ss1)
    [trainacc,traintime,xx1,yy1,par1,con0,train_y,tn] = multi_scale_bls_breath(train_x,train_y,test_x,test_y,s,C,h,fo,f0,par1,[],1,N1,N2,fe,fe1,[],nfp,[]);
%     leng0=lengo(ss1);
%     leng0=[];
%     for i=1:length(leng01)
%         leng0=[leng0 leng01{i}];
%     end
        leng0=[];
        for lp=1:length(ss1)
            leng0=[leng0 lengo(ss1(lp))];
        end
    for xi=1:length(leng0)
        K0(xi)=sum(leng0(1:xi));
    end
    K0=[0 K0];
    [testacc,testtime,x1,y1,par1,~,~] = multi_scale_bls_breath(train_x,train_y,test_x,test_y,s,C,h,fo,f0,par1,[],2,N1,N2,fe,fe1,con0,nfp,K0);
    clear K0
    trainacc=trainacc(end);
end

function [tr0,nmark]=sub_breath(tr,con,ss,leng,threshold0)
leng=leng(ss);
for i=1:length(ss)
    k(i) = sum(leng(1:i));
end
k=[0 k];tr0=[];n=0;nmark=0;
for i=1:length(leng)
    tr1=tr(k(i)+1:k(i+1));
    con1=con(k(i)+1:k(i+1));
    h=length(find(tr1==1))-length(find(con1<1));
    threshold=threshold0*length(tr1)/60*2;
    if h>threshold
        n=n+1;
        nmark(n)=i;
        mk=1;
    else
        mk=eps;
    end
    tr0=[tr0;mk*ones(length(tr1),1)];
end
if nmark~=0
   nmark=ss(nmark);
end
end

function [FF0,f,fe,f0,nfp]=generate_exfeature(FF,time_related,Mode,rep,fo1,para,leng,olen,kl)
ss=1:length(leng);
nrep=rep{2};nfo1=fo1{2};
rep=rep([1 4 ]);fo1=fo1([1 4 ]);Mode=Mode([1 4 ]);%3
FF0=[];f=[];M=[];
cen=ceil(length(para)/2);

for i=1:length(rep)
    time_related1=time_related(i);
    nex=time_related1*2+1;
    Nex(i)=nex;
    m=rep{i};fo=fo1{i};mode=Mode(i);
    FF2=FF(:,m);
    para1=para(cen-time_related1:cen+time_related1);
    [FF1,fex]=matexpand(FF2,time_related1,fo,para1,mode,ss,leng);
    FF0=[FF0 FF1];
    %     train_x2=train_x(:,m);test_x2=test_x(:,m);
    %     [train_x1,fex]=matexpand(train_x2,time_related,fo,para,mode,ss,leng);%需区分分步扩展的和整体扩展的
    %     [test_x1,~]=matexpand(test_x2,time_related,fo,para,mode,ss1,leng);
    %     train_x0=[train_x0 train_x1];test_x0=[test_x0 test_x1];
    f=[f fex];
    fsum(i)=length(f);
    M=[M m];
end
[fe,f0,nfp]=gefeaenhan(f,fsum,Nex(2),olen,kl,2);
fe=[fe];
M1=setdiff(1:size(FF,2),M);
FF0=[FF0 FF(:,M1)];
end

function     [xx,yy,x,y,par,leen,trainacc,testacc,tn,train_y]=featurebasedbls(f1,f,fe,f0,nfp,train_x,test_x,train_y,test_y,...
    train_x0,test_x0,leng,s,C,h,par,pare,N1,N2,fe0,fe1,ss,ss1)
train_x=[train_x0 train_x]; test_x=[test_x0 test_x];
[~,leen]=size(train_x);
f=[f f1];

K0=[];
[trainacc,traintime,xx,yy,par,con0,train_y,tn] = multi_scale_bls41(train_x,train_y,test_x,test_y,s,C,h,f,f0,par,pare,1,N1,N2,fe,fe1,[],nfp,[]);
%     [trainacc,traintime,xx,yy,par,con0,train_y,tn] = recurent_BLS1(train_x,train_y,test_x,test_y,s,C,h,f,f0,par,1,N1,N2,fe,0,nfp,[],1);
leng0=[];
for lp=1:length(ss1)
    leng0=[leng0 leng(ss1(lp))];
end
for xi=1:length(leng0)
    K0(xi)=sum(leng0(1:xi));
end
K0=[0 K0];
[testacc,testtime,x,y,par,~,~] = multi_scale_bls41(train_x,train_y,test_x,test_y,s,C,h,f,f0,par,[],2,N1,N2,fe,fe1,con0,nfp,K0);
%     [testacc,testtime,x,y,par,~,~,~] = recurent_BLS1(train_x,train_y,test_x,test_y,s,C,h,f,f0,par,2,N1,N2,fe,con0,nfp,K0,1);
clear K0
trainacc=trainacc(end);
end

function [C_xr]=xexpan(x,time_related,ss,leng,mode)
gg1=1;gg2=0;
for i=1:length(ss)
    gg=leng((ss(i)));
    gg2=gg2+gg;
    C_xr(gg1:gg2,:)=time_causal(x(gg1:gg2,:),time_related,mode);
    gg1=gg1+gg;
end
end

function [xx1,yy1,x1,y1,par1,trainacc,testacc]=expandebls(train_x1,test_x1,train_y,test_y,xx,x,time_related,s,C,h,f0,f00,par,pare,N1,N2,fe,fe1,nfp,leng,ss,ss1,train_xb,test_xb,fb,fb0,fbe)
para=[0.6 0.6 0.7 0.7 0.7 0.7 0.75 0.8 0.9 0.95 1 1 0.9 0.85 0.8 0.75 0.7 0.6 0.6 0.6 0.6 0.6 0.6];
sta=(length(para)-(time_related*2)-1)/2;
para=para(sta+1:end-sta);
para=repmat(para,1,5);
% p=[];
% for i=1:length(para)
%     p=[p para(i)*ones(1,5)];
% end
% para=p;
    xtr=time_causal(xx,time_related,2);
    xtr=xtr.*para;
    train_x1=[train_x1 xtr];
    xte=time_causal(x,time_related,2);
    xte=xte.*para;
    test_x1=[test_x1 xte];
%     [train_x1, test_x1]=pre_zca(train_x1,test_x1);
    nex=time_related*2+1;
    [fk,fk0,~]=blstimekind(size(train_x1,2),f0);
% %%%%%%呼吸
% %nfp在外层循环已经计算过，因此不需要重新计算，此处仅仅增加对f和f0的计算
    train_x1=[train_x1 train_xb];
    test_x1=[test_x1 test_xb];
%     fk=[f0 nex*ones(1,size(xtr,2)/nex)];
%     fk0=[f00 sum(f0):sum(fk)];
%     fb0=sum(fk)+1:sum(fk)+sum(fb);
    for i=1:length(fbe)
        fbe1=cell2mat(fbe(i))+sum(fk);
        fbe0(i)={fbe1};
    end
    fe=[fe;fbe0'];
    fk=[fk fb];
    fk0=[fk0 fb0];
%%%%%
    [trainacc,traintime,xx1,yy1,par1,con0,train_y,tn] = multi_scale_bls41(train_x1,train_y,test_x1,test_y,s,C,h,fk,fk0,par,[],1,N1,N2,fe,fe1,[],nfp,[]);
        leng0=[];
        for lp=1:length(ss1)
            leng0=[leng0 leng(ss1(lp))];
        end
    for xi=1:length(leng0)
        K0(xi)=sum(leng0(1:xi));
    end
    K0=[0 K0];
    [testacc,testtime,x1,y1,par1,~,~] = multi_scale_bls41(train_x1,train_y,test_x1,test_y,s,C,h,fk,fk0,par1,[],2,N1,N2,fe,fe1,con0,nfp,K0);
    clear K0
    trainacc=trainacc(end);
end

function [train_x1,train_x2,test_x1,test_x2]=geset(train_x,test_x,f,rep)
% f=f(1:end-1);rep=setdiff(rep,length(f));
% train_x0=train_x(:,end);test_x0=test_x(:,end);
fk=setdiff(1:length(f),rep);
for i=1:length(f)
    k(i)=sum(f(1:i));
end
k1=ones(1,length(k));
k1(2:end)=k1(2:end)+k(1:end-1);
f1=[];f2=[];
for i=1:length(rep)
    f1=[f1 k1(rep(i)):k(rep(i))];
end
for i=1:length(fk)
    f2=[f2 k1(fk(i)):k(fk(i))];
end
train_x1=[train_x(:,f1) ];
train_x2=[train_x(:,f2) ];
test_x1=[test_x(:,f1) ];
test_x2=[test_x(:,f2) ];
end

function [ACC_all_1,ACC_part_1,Accuracy_1,Precidion_1,Recall_1,Specificity_1,F1SCORE_1,Kappa_1,Mcc_1]=resultstatsic(y,test_y,n,j)
[Acc_all_1,Acc_part_1,accuracy_1,precision_1,recall_1,specificity_1,F1score_1,kappa_1,mcc_1] = Evaluation(y,test_y,n);
ACC_all_1=Acc_all_1;%ACC是总体百分比
ACC_part_1=Acc_part_1;%ACC每一项是百分比
Accuracy_1=accuracy_1;
Precidion_1=precision_1;
Recall_1=recall_1;
F1SCORE_1=F1score_1;
Kappa_1=kappa_1;
Mcc_1=mcc_1;
Specificity_1=specificity_1;
end

function fe=othersignal(da)
da1=da(da<quantile(da,0.9));
me1=mean(da1);
me=mean(da);
if me/me1>6
    da=da1;
    da(da>quantile(da,0.9))=quantile(da,0.9);
end
da=da-mean(da);
s=diff(da>0);
l1=length(find(s==1));
l2=length(find(s==-1));
zc=l1+l2;
st=std(da1);
mea=median(da1);
fe=[me mea st zc];
end


function [train_x,test_x,endpoint]=retrainsetosalabel(kk,train_x,test_x,leen,xr,xe,leng,ss,ss1)
endpoint=4;
lengr=leng(ss);lenge=leng(ss1);
xr1=[];xe1=[];
for i=1:size(xr,1)
    xr1=[xr1 repmat(xr(i,:),lengr(i),1)];
end
for i=1:size(xe,1)
    xe1=[xe1 repmat(xe(i,:),lenge(i),1)];
end


if kk==1
    train_x(:,leen+1:leen+endpoint)=xtr;
    test_x(:,leen+1:leen+endpoint)=xte;
else
    train_x(:,leen+1:leen+endpoint)=(train_x(:,leen+1:leen+endpoint)+xtr)/2;
    test_x(:,leen+1:leen+endpoint)=(test_x(:,leen+1:leen+endpoint)+xte)/2;
end

end


function [train_x,test_x,endpoint]=retrainsetosa(kk,train_x,test_x,leen,x_tr,x_te,tl,mode,cr1,ce1)
nl=size(x_tr,2);
endpoint=tl*2*nl+nl;
% a1=0.15;a2=a1-0.1;
% s=1:-a1:1-a1*tl;
% if mode==1
%     s=sort(s);
% else
%     s1=sort(s)+a2;
%     s=[s1(1:end-1) s];
% end
% ls=endpoint/5;
% s0=[];
% 
% s0=repmat(s,1,tl*mode+1);
xtr=time_causal(zscore((x_tr.*cr1)')',tl,mode);
xte=time_causal(zscore((x_te.*ce1)')',tl,mode);
% xtr=xtr.*s0;
% xte=xte.*s0;

if kk==1
    train_x(:,leen+1:leen+endpoint)=xtr;
    test_x(:,leen+1:leen+endpoint)=xte;
else
    train_x(:,leen+1:leen+endpoint)=(train_x(:,leen+1:leen+endpoint)+xtr)/2;
    test_x(:,leen+1:leen+endpoint)=(test_x(:,leen+1:leen+endpoint)+xte)/2;
end

end


function [train_x,test_x,endpoint]=retrainset(kk,train_x,test_x,leen,x_tr,x_te,tl,mode,cr1,ce1)
% tl=2;
% mode=2;
endpoint=tl*mode*5+5;
% corra=[1 1 1 1 1];
a1=0.15;a2=a1-0.1;
s=1:-a1:1-a1*tl;
if mode==1
    s=sort(s);
else
    s1=sort(s)+a2;
    s=[s1(1:end-1) s];
end
ls=endpoint/5;
s0=[];
% for i=1:ls
%     s0=[s0 s(i)*ones(1,5)];
% end
s0=repmat(s,1,tl*mode+1);
xtr=time_causal(zscore((x_tr.*cr1)')',tl,mode);
xte=time_causal(zscore((x_te.*ce1)')',tl,mode);
xtr=xtr.*s0;
xte=xte.*s0;
% if mo==1
%     train_x=[train_x xtr];
%     test_x=[test_x xte];
% else
if kk==1
    train_x(:,leen+1:leen+endpoint)=xtr;
    test_x(:,leen+1:leen+endpoint)=xte;
else
    train_x(:,leen+1:leen+endpoint)=(train_x(:,leen+1:leen+endpoint)+xtr)/2;
    test_x(:,leen+1:leen+endpoint)=(test_x(:,leen+1:leen+endpoint)+xte)/2;
end

end

function [fe,f0,nfp]=gefeaenhan(fo,fsum,m,olen,kl,t)
%仅针对ft进行多导信号同步  olen原始眼电通道长度  kl眼电单位系数  t需要扩展的特征种类
n=olen/(6/kl);%特征种类数
f=[fsum(t-1) fsum(t)];
for i=1:length(fo)
    fs(i)=sum(fo(1:i));
end
f=fs(f(1))+1:fs(f(2));
bu=length(f)/m/olen;%bu<2 EEG+EOG kl>1     bu==2 EEG+EOG  kl==1   or   EEG+EOG+EMG  kl==2
%bu==3 EEG*2 EOG*2  kl==2

fet=f(1:m*olen*kl);
fot=f(m*olen*kl+1:m*olen*(kl+1));
% if bu>=3
%    fmt=f(m*olen*(kl+1)+1:end);
% end
d=1;%取多长时间的脑电眼电同步信号，取1为10s，取2为20s，取3为30s
olp=rem(6,d*kl);
for i=1:olen*m/d
    % ften0=[fet(6*(i-1)+1:6*i) fot(6/kl*(i-1)+1:6/kl*i)];
    fet1=fet(d*kl*(i-1)+1 - (i-1)*olp : d*kl*i - (i-1)*olp);
    fot1=fot(d*(i-1)+1:d*i);
    fe{i}=[fet1 fot1];
end
% % f0=[6:14 64:84]+(m-1)/2*sum(fo(1:fsum(1)))/m;

f0=[12:18 221:247]+(m-1)/2*sum(fo(1:fsum(1)))/m;
nfp=length(fo)-length(fo);
end

function [f,f0,fe,nfp]=blstimekindosa(len,f1,fe)
f=[f1 2*ones(1,(len-sum(f1))/2)];%增加的 c
if length(f)>len
    f0=[1:5 10:15 sum(f1):len];
else
    f0=[1:5 10:15];
end
    nfp=(len-sum(f1))/2;
    fe=[fe;sum(f1):len];
end


function [Label,leng,X]=increasesstages(label,stage,x,mark,leng,si)
len1=[];X=[];Label=[];l1=1;l2=0;
for i=1:length(si)
    le=leng(si(i));
    l2=l2+le;
    label1=label(l1:l2);
    x1=x(l1:l2,:);
    [label1,h0,len,s]=increasedata(label1,stage);
    if ~isempty(s)
        x1=fixdata(x1,h0,s,mark);
    end
    Label=[Label;label1];
    len1=[len1 len];
    X=[X;x1];
    clear x1 len label1
    l1=l1+le;
end
leng(si)=len1;
end

function [train_x,train_y,train_yy,leng1,dimarks]=samplebalance(train_x,train_yy,len1,leng,ss)
mark=setdiff(1:size(train_x,2),[80:100]);
%     mark=1:size(train_x,2);
train_x=[train_x zeros(len1,1)];
leng1=leng;
m1=[1 3 5];
m2=[0.05 0.15 0.15];
k=0;
for i=1:length(m1)
    if length(find(train_yy==m1(i)))/length(train_yy)<m2(i)
        k=k+1;
        mk(k)=m1(i);
    end
end
if ~isempty(mk)
    for i=1:length(mk)
        [train_yy,leng1,train_x]=increasesstages(train_yy,mk(i),train_x,mark,leng1,ss);
    end
end
dimark=find(train_x(:,end)==1);
dimarks=setdiff(1:length(train_x),dimark);
train_x=train_x(:,1:end-1);
train_y=label2lab(train_yy);
end

function [X1,X2,LABEL,leng1]=banwake(X,X0,Label,leng,b)
l1=1;l2=0;X1=[];X2=[];LABEL=[];leng1=[];
for i=1:length(leng)
    le=leng(i);
    l2=l2+le;
    x=X(l1:l2,:);
    x1=X0(l1:l2,:);
    label=Label(l1:l2);
    n1=length(find(label==2))/length(label);
    n2=length(find(label==5))/length(label);
    if n2>n1 || n2>0.5
        m1=find(label~=5,1);
        m2=find(label~=5,1,'last');
        %         m=find(label==5);
        %         s=randperm(length(m));
        %         s1=s(1:floor(length(s)/5));
        s1=1:ceil(m1*b);
        s20=length(label)-m2;
        s2=floor(m2+s20*(1-b)):length(label);
        dimark=setdiff(1:size(x,1),union(s1,s2));
        x=x(dimark,:);
        x1=x1(dimark,:);
        label=label(dimark);
    end
    len1=length(label);
    X1=[X1;x];
    X2=[X2;x1];
    LABEL=[LABEL;label];
    leng1=[leng1;len1];
    l1=l1+le;
end
end

function [train_y00,study_rate,si]=trainybanlance1(kk,train_x,train_yy,train_y,study_rate,xx,yy,cr1,train_y0,con_st,si,ss,ko0)
train_y00=[];
for bo=1:length(ss)
    trainy=train_y(ko0(bo)+1:ko0(bo+1),:);
    trainx=train_x(ko0(bo)+1:ko0(bo+1),:);
    trainyy=train_yy(ko0(bo)+1:ko0(bo+1));
    if size(train_y0,1)==size(train_yy,1)
        trainy0=train_y0(ko0(bo)+1:ko0(bo+1));
    else
        trainy0=train_y0;
    end
    xx0=xx(ko0(bo)+1:ko0(bo+1),:);
    yy0=yy(ko0(bo)+1:ko0(bo+1),:);
    cr10=cr1(ko0(bo)+1:ko0(bo+1));
    [trainy,study_rate,si]=trainybanlance(kk,trainx,trainyy,trainy,study_rate,xx0,yy0,cr10,trainy0,con_st,si);
    train_y00=[train_y00;trainy];
end
end

