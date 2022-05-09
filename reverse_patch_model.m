%----------------------------------------%
% Transforming Patch model to original image%
%                                        Date:2015.03                     %
% Revised in 2016.06
%Author: Xiaoyang Wang
%----------------------------------------%

function [Original]=reverse_patch_model(Patch,m,n,p,q,endRow,endColumn)       % m & n shows the number of rows and columns of one patch
[i,j]=size(Patch);
if i~=m*n
    disp('Error! input m or n is not correct');
end 
N=endColumn/n;          
New=zeros(endRow,endColumn);
for ii=1:j
    t=rem(ii,N);
    if t==0
        t=N;
    end
    New((ceil(ii/N)-1)*m+1:ceil(ii/N)*m,((t-1)*n+1):t*n)=reshape(Patch(:,ii),m,n);
end
Original=New(1:p,1:q);
end