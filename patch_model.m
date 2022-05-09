%----------------------------------------%
% Build infrared patch model with non-overlap%
%                                        Date:2015.03                     %
% Revised in 2016.06
%----------------------------------------%

function [New,endRow,endColumn]=patch_model(I,m,n)       % m & n shows the number of rows and columns of one patch
I=double(I);
[p q]=size(I);
endRow=ceil(p/m)*m;
endColumn=ceil(q/n)*n;

Up=[I,repmat(I(:,q),1,endColumn-q)];
newI=[Up;repmat(Up(m,:),endRow-m,1)];

T=(endRow*endColumn)/(m*n);      %image patch amount 
N=endColumn/n;          
% New=zeros(m*n,T);
New=zeros(m,n,T);
for ii=1:T
    t=rem(ii,N);
    if t==0
        t=N;
    end
%     New(:,ii)=reshape(newI((ceil(ii/N)-1)*m+1:ceil(ii/N)*m,((t-1)*n+1):t*n),m*n,1);
    New(:,:,ii)=reshape(newI((ceil(ii/N)-1)*m+1:ceil(ii/N)*m,((t-1)*n+1):t*n),m,n);
end
end
