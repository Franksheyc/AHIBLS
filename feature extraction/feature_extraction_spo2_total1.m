warning off
clear
clc
bx=[1 2 3 4 7];%therm press abdo thor pleth
samplingrate1=32;
samplingrate2=128;
fx1=0.1;fx2=4;
fx11=0.05;fx22=0.4;
[BT_b_B,BT_b_A]=filterdesign(fx11,fx22,samplingrate1);
[BT_c_B,BT_c_A]=filterdesign(fx1,fx2,samplingrate2);
[BT_o_B,BT_o_A]=filterdesign(0.014,0.033,1);

set={'ccshs';'cfs';'mesa';'sof';'shhs';'mros'};
% channel={{'ECG1','AIRFLOW','NASALPRES','ABDOEFFORT','THOREFFORT','SpO2','POSITION','PlethWV'};
%     {'ECG1','AIRFLOW','NASALPRES','ABDOEFFORT','THOREFFORT','SpO2','POSITION','PlethWV'};
%     {'EKG','Therm','Flow','Abdo','Thor','SpO2','Pos','Pleth'};
%     {'ECG1','Airflow','Cannula Flow','Abdominal','Thoracic','SAO2','Position'};
%     {'ECGL','Airflow','CannulaFlow','ABD','Chest','SpO2','Position'}};% visit1 'Abdominal','Thoracic','SAO2'
channel={'ecg';'therm';'press';'abdo';'thor';'spo2';'pos';'pleth'};

ad='\\192.168.31.100\Data\13_公开数据集\公开数据集\新建文件夹\新建文件夹';
ad2='D:\database\data\label';
ad1='D:\database\data\OSA features\';
n=dir(ad);
sk=[2 3 4 5];
spo2feature=[];spo2feature_nowake=[];spo2con=[];
for i=1:length(sk)%5:length(n)-3
    na=set{sk(i)};%n(sk(i)+2).name;
%     [~,p]=max(strcmp(na,set));
%     channel1=channel(p);
    n1=[ad,'\',na];
%     if i==5 &&  length(nc)>
%     n2=[ad1,'\',na,'1'];
%     end
%     mkdir(n2);
    nc=dir(n1);
    n21=[ad2,'\',na,'\label'];
    nc1=dir(n21);
    load([n1,'\total.mat'])
    for j=1:length(nc)-3
        tic
        if sk(i)~=5
            n2=[ad1,'\',na];
        end
        if sk(i)==5 && j<=5793
            n2=[ad1,'\',na,'1'];
        elseif sk(i)==5 && j>5793
            n2=[ad1,'\',na,'2'];
        end
        mkdir(n2);
        nodata=0;%nodatacha{j};
        na1=nc(j+2).name;
        filename=[n1,'\',na1];
        load(filename,'spo2','head','rate')
        

        filename1=[n21,'\',nc1(j+2).name];
        load(filename1)
        label1=zeros(1,length(spo2));
        for kl=1:length(label)
            label1(30*(kl-1)+1:kl*30)=label(kl);
        end

        j
        con1=0;con2=0;con3=0;con4=0;
        le=head.records;len=floor(le/30);le=len*30;
        
        if ~ismember(6,nodata)
            fs_spo2=floor(length(spo2)/le);
            for k=1:len
                s=spo2((k-1)*fs_spo2+1:k*fs_spo2);
                spo2con1(k)=mean(s)>80;
            end
            spo2(spo2<50)=50;

            con1=1;
            spo2=spo2(label1~=0);
            [fea1]=spo2fea(spo2,fs_spo2,len,BT_o_B,BT_o_A);
        end
%         spo2feature=[spo2feature;fea];
%         spo2con=[spo2con spo2con1];
        spo2feature_nowake=[spo2feature_nowake;fea1];
        runtime(j)=toc
    end
    runtime=mean(runtime);
    savename1=[n2,'\spo2feature_total.mat'];
    save(savename1,'spo2feature_nowake','spo2con','runtime')% check 5s  check1 10s
    clear spo2feature spo2con spo2feature_nowake spo2
end

function breathfeature=breathf(len,segp,pp_p,peaks_p,troughs_p)
for i=1:len
    markp=find(segp==i);
    if ~isempty(markp)
        ppp=pp_p(markp);
        pkp=peaks_p(markp);
        ptp=troughs_p(markp);
        fp(i)=length(markp)/30;
    else
        ppp=0;
        pkp=0;
        ptp=0;
        fp(i)=0;
    end
    pppm1(i)=max(ppp);
    pppm2(i)=mean(ppp);
    pppm3(i)=min(ppp);
    pppm4(i)=std(ppp);
    pkpm1(i)=max(pkp);
    pkpm2(i)=mean(pkp);
    pkpm3(i)=min(pkp);
    pkpm4(i)=std(pkp);
    ptpm1(i)=max(ptp);
    ptpm2(i)=mean(ptp);
    ptpm3(i)=min(ptp);
    ptpm4(i)=std(ptp);
end
breathfeature=[pppm1' pppm2' pppm3' pppm4' pkpm1' pkpm2' pkpm3' pkpm4' ptpm1' ptpm2' ptpm3' ptpm4' fp'];
end

function [spo2feature,spo2con,spo2con1]=spo2fea(spo2,fs,len,bb,ba)
sp=find(spo2>60);
spo2=spo2(sp);
spo2=spo2/100;
% dl=length(spo2)-length(sp);
%%
means=mean(spo2);
medians=median(spo2);
mins=min(spo2);
stds=std(spo2);
Mm=max(spo2)-min(spo2);% range

px=length(find(spo2<medians))/length(spo2);% px

spo2m=spo2>=means;
zc=length(find(spo2m==0))/length(spo2); %zc

w=12;
for i=1:floor(length(spo2)/w)
s(i)=mean(spo2((i-1)*w+1:i*w));
end
dels=mean(diff(s));% delta x

%% Entropy
spo2=spo2*100;
[apEn,~] = getApEn(spo2,'m',1,'r',0.25*stds,'tau',300);
[A,~] = DFA_fun(spo2,300,2);
pep=PermutationEntropy( spo2 , 30, 4, 300); % 5min win  观察变化
pep=mean(pep);
pes = sampleEntropy(spo2, 1, 0.25, 1);
[lzc, ~,~] = calc_lz_complexity(spo2<mean(spo2), 'primitive', 1);
apen=apEn;
dfa=A(1);

for i=1:length(spo2)-2
    h(i)=sqrt((spo2(i+2)-spo2(i+1))^2+(spo2(i+1)-spo2(i))^2 );
end
h1=h<0.25;% r=0.25
if isempty(h1)
   h1=0;
end
CTM=length(find(h1==1))/(length(spo2)-2);% CTM_r
%%
spo2d=diff(spo2);
s=find(spo2d<0);
s=s(intersect(find(s>2),find(s<length(spo2))));
d=10;
for i=1:length(s)
    s1s=spo2(max(s(i)-d,1));
    s1e=spo2(min(s(i)+d,length(spo2)));
    s1=spo2(max(s(i)-d,1):min(s(i)+d,length(spo2)));
    s1=s1/100;
    prsa(i)=(spo2(s(i))+spo2(s(i)+1)-spo2(s(i)-1)-spo2(s(i)-2))/4;
    pMm(i)=max(s1)-min(s1);
    slope(i)=(s1(end)-s1(1))/length(s1);
    slope1(i)=(spo2(s(i))-s1s)/min(10,s(i)-1);      %PRSA slope
    slope2(i)=(spo2(s(i))-s1e)/min(10,length(spo2)-s(i));
end
prsadc=mean(prsa);
prsadad=mean(pMm);
prsaws=mean(slope);
prsasb=mean(slope1);
prsasa=mean(slope2);

%%
% spo2f=abs(fft(spo2,1024));
% spo2p=sum(spo2f(2:512));% psd total
spo2=spo2/100;
spo2p=pwelch(spo2,300,0,512);
spo2p=sum(spo2p(2:end));% psd total
spo2b=filtfilt(bb,ba,spo2);
spo2b=pwelch(spo2b,300,0,512);
% spo2bp=findpeaks(spo2b);
spo2b=sum(spo2b(2:end));% psd band
spo2r=spo2b/spo2p;% psd ratio
spo2bp=max(spo2b);% psd band peak
spo2p=spo2p/100;
%% Oxigen Desaturation event
baseline=min(0.95,mean(spo2));
sp1=find(spo2<=baseline);
sp1=intersect(sp1,21:length(spo2)-20);
sp1=[0 sp1];
sp2=diff(sp1);
sp3=find(sp2>1);
sp3=[sp3 length(sp1)];

mj=0; slope=0;% od slope
ox=0; ox100=0; area=0;
area100=0; ac=0; area1=0; t1=0; time=0;
for i=1:length(sp3)-1
    sp=sp1(sp3(i)+1:sp3(i+1));
    a0=max(sp(1)-10,1);
    [~,b]=min(spo2(sp));
    d0=min(sp(end)+10,length(spo2));
    sa0=spo2(a0:sp(1));
    
    for j=5:-1:1
        sa01=mean(sa0(1:end-(j)));
        if sa01-sa0(end-j-1)>=0.01
            a=sp(1)-j;
            break
        else
            a=sp(1);
        end
    end
    sd0=spo2(sp(end):d0);
    for j=1:3
        sd01=mean(sd0(j+1:end));
        if sd01-sd0(end-(j-1))<=0.01
            d=sp(end)+j;
            break
        else
            d=sp(end);
        end
    end

    sa=sa01;
    sb=min(spo2(sp));
%     sd=spo2(d);
%     if sd-sb<3 && spo2(d+1)-b>3
%        d=d+ceil((spo2(d+1)-sd)/3);
%     end

    if d-a<10 || d-a>90 || sa-sb<0.03
       continue
    else
        mj=mj+1;
        slope(mj)=(sb-sa)/(b-a);% od slope
        ox(mj)=sa-sb; 
        ox100(mj)=1-sb;
        area(mj)=sum(baseline-spo2(sp));
        area100(mj)=sum(1-spo2(sp));
        ac=0.9-spo2(sp);
        area1(mj)=sum(ac(ac>0));
        t1(mj)=length(ac>0);
        time(mj)=d-a+1;
    end
end
ODX=mj/(length(spo2)/3600);
dlmu=mean(time);
dlst=std(time);
dlmax=mean(ox);
dlmaxst=std(ox);
dl100=mean(ox100);
dl100st=std(ox100);
dls=mean(slope);
dlsst=std(slope);
dla=mean(area);
dlast=std(area);
dla100=mean(area100);
dla100st=std(area100);
if length(time)<2
   dtime=time;
else
    dtime=diff(time);
end
dldt=mean(dtime);
dldst=std(dtime);
%%
POD=dlmu/(length(spo2)/60);
AODmax=dla/(length(spo2)/3600);
AOD100=dla100/(length(spo2)/3600);
CT=sum(time)/(length(spo2));
CA=sum(area1)/(length(spo2));

spo2feature=[means medians mins stds Mm px zc dels  ...    %1:8
    pep pes apen dfa lzc CTM  ...                          %9:14
    prsadc prsadad prsaws prsasb prsasa...                 %15:19
    spo2p spo2b spo2bp spo2r...                            %20:23
    ODX dlmu dlst dlmax dlmaxst dl100 dl100st dls dlsst... %24:32
    dla dlast dla100 dla100st dldt dldst POD AODmax AOD100 CT CA];  %33:43
%%

%     f1(i,:)=persentage(s,0.95);
%     f2(i,:)=persentage(s,0.9);
%     f3(i,:)=persentage(s,0.85);
%     f4(i,:)=persentage(s,0.8);


    


% spo2feature=[mdss1' mdss2' m1' m2' m3' sts' low' low1' sk' ku' mds1' mds2' mds3' pe0 pe1 pe2 pe3 rms...
%        f1 f2 f3 f4 ];
% useless=sum(isnan(spo2feature)')';
% useless=useless+(f4(:,2)>0.5);
% spo2feature=[spo2feature useless];

end

function [Pw,pe10,pe20,pe30]=entropys(s)
    Pw=s./sum(s);
    ls=u0-length(s);
    s0=[s repmat(s(end),1,ls)];
    pe10 = PermutationEntropy( s0 , floor(length(s0)/8), 4, floor(length(s0)/2));
    pe20 = PermutationEntropy( s0 , floor(length(s0)/6), 4, floor(length(s0)/3));
    pe30 = sampleEntropy(s, 10, max(0.011,max(diff(s))), 1);%对spo2，用r=0.01

end

function f=persentage(s,n)
s1=s>n;
pp=length(find(s1==1))/length(s1)/2;
np=length(find(s1==0))/length(s1)/2;
f=[pp np];
% s1=diff(s1);
% ss=find(s1~=0);
% zc=length(ss);
% f=[pp np zc];
%pp 大于某阈值的比例
%np 小于某阈值的比例
%zc 在某阈值上下变化次数
end

function sigcon=sigcheck(sig,fs,len,dma)
for i=1:len
    s=sig((i-1)*fs*30+1:i*fs*30);
    s1=findpeaks(s,'MinPeakHeight',0);
    sigcon(i)=dma>mean(s1)+3*std(s1);
end
end


function [BT_SW_B,BT_SW_A]=filterdesign(fx1,fx2,samplingrate)
SW = [fx1 fx2] / samplingrate * 2;
[BT_SW_B,BT_SW_A] = butter( 4 , SW , 'bandpass' ); %设计巴特沃斯滤波器参数
end

function [dma]=dimaxbreath(sig)
d1=quantile(sig,0.95);
d2=mean(sig)+3*std(sig);
if d2>d1
    dma=mean(sig)+5*std(sig);
else
    dma=d1;
end
end

function timefeatures=bfeature(DAT,timepiece,window,overlap)
for i=1:timepiece    
    dat=DAT(:,(i-1)*window-(i-1)*overlap+1:i*window-(i-1)*overlap);
    means=mean(dat')';
    stds=std(dat')';
    Eng=sqrt(sum(dat'.^2))';
    sk=skewness(dat')';
    ku=kurtosis(dat')';
    med = median(dat')';%每一段中值
    lrssv=log10(sqrt(sum((diff(dat').^2))))';
    pw=abs(dat)'./sum(abs(dat)');
    pe = -sum(pw.*log(pw))';
    z0=dat>0;
    z=diff(z0')';
    for j=1:t
        zc(j,:)=length(union(find(z(j,:)==1),find(z(j,:)==-1)))/window;
        pe10 = PermutationEntropy( dat(j,:) , floor(window/8), 4, floor(window/2));
        pe1(j,:)=mean(pe10')'; 
    end
    
    
    timefeatures.means(:,i)=means;
    timefeatures.stds(:,i)=stds;
    timefeatures.ku(:,i)=ku;
    timefeatures.sk(:,i)=sk;
    timefeatures.Eng(:,i)=Eng;
    timefeatures.Pe(:,i)=pe;
    timefeatures.Z(:,i)=zc;
    timefeatures.median(:,i)=med;
    timefeatures.Pe1(:,i)=pe1;
    timefeatures.LRSSV(:,i)=lrssv;
end
end

function hrvfeature

end

function ppgfeature

end