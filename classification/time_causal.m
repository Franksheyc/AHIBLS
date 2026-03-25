%poi位置系数矩阵，长度为2*time_related+1   mode 整体（1）或分部（2）扩展
function yy=time_causal(lab,time_related,mode)
[a,b]=size(lab);%分类数
if b>a
    lab=lab';
end
[a,b]=size(lab);
% yy=zeros(length(lab),(time_related+1)*c);
% yy(:,time_related*c+1:(time_related+1)*c)=lab;
%    a=time_related+1;
%    b=length(lab);
%    for i=1:time_related
%        for j=1:time_related
%        yy(i,c*(j-1)+1:c*j)=0;
%        end
%        yy(a+1-i:end,c*(i-1)+1:c*i)=lab(1:end-a+i,:);
%    end
%    for i=1:time_related
%        for j=1:time_related
%        yy(b+1-i,c*(a+j-1)+1:c*(a+j))=0;
%        end
%        yy(1:b-a+i,c*(a+i-1)+1:c*(a+i))=lab(a-i+1:end,:);
%    end
%    for i=1:time_related/2
%        yy1=yy(:,c*(i-1)+1+a*c:c*i+a*c);
%        yy(:,c*(i-1)+1+a*c:c*i+a*c)=yy(:,a*c+c*(a-i-1)+1:c*(a-i)+a*c);
%        yy(:,a*c+c*(a-i-1)+1:c*(a-i)+a*c)=yy1;
%    end
   m=2*time_related+1;
   yy=zeros(a+m-1,m*b);
   y=zeros(a+m-1,1);
   
   for j=1:b
      for i=1:m
          y(m-i+1:a+m-i,i)=lab(:,j);
      end
      yy(:,(j-1)*m+1:j*m)=y;
   end

   if mode==1
      for i=1:m
          y1=zeros(a+m-1,b);
          y1(m-i+1:a+m-i,:)=lab;
          yy(:,(i-1)*b+1:i*b)=y1;
      end
   end
   yy=yy(time_related+1:time_related+length(lab),:);
   

%    if mode==1
%        yy0=[];
%        for i=1:b
%            yy1=yy(:,(i-1)*m+1:(i-1)*m+time_related+1);%只取之前的label
%            yy0=[yy0 yy1];
%        end
%        yy=yy0;
%    end
end