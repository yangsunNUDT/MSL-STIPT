%% Stable multi-subspace learning and spatial-temporal tensor model for infrared small target detection
%% V1 version of MSLSTIPT, which uses sliding windows to construct the spatial-temporal infrared tensor, and it is applied to high-frequency sequence
%% We also paorposed V2 version of MSLSTIPT, which can also be applied for low-frequency infrared sequence
clc;
clear;
close all;

addpath(genpath('proximal_operator'));
addpath(genpath('tSVD'));
addpath(genpath('WSNM_problem'));

% setup parameters
C=1e-4;
% C=10;
p=0.8;
num_image=6;%L
% set patch size (adjustable)
m = 100;
n = 100;
%% input data
strDir='data\';
tic
k=6;% total number of images
picname=[strDir  num2str(1),'.bmp'];
I=imread(picname);
    [mm, nn, ch]=size(I);
    if ch==3
        I=rgb2gray(I);
    end
    [tempD,endRow,endColumn]=patch_model(I,m,n);
    [~,~,onesize]=size(tempD);
    D=zeros(m,n,onesize*k);
for i=1:k
    fprintf('%d/%d: %s\n', 120, i);
    picname=[strDir  num2str(i),'.bmp'];
    I=imread(picname);
    [~, ~, ch]=size(I);
    if ch==3
        I=rgb2gray(I);
    end
    % construct spatial-tensor patch model
    [tempD,endRow,endColumn]=patch_model(I,m,n);    
    D(:,:,(i-1)*onesize+1:onesize*i)=tempD;
end
      tenD=double(D);
      tenD =tenD/255;% nomalize
        size_D=size(tenD);
        [n1,n2,n3]=size(tenD);
        n_1=max(n1,n2);%n(1)
        n_2=min(n1,n2);%n(2)
        patch_frames=num_image*onesize;%
        patch_num=n3/patch_frames;
for l=1:patch_num
    temp=zeros(n1,n2,patch_frames);
    for i=1:patch_frames
        temp(:,:,i)=tenD(:,:,patch_frames*(l-1)+i);
    end     

    %% test R-TPCA
    temp=reshape(temp,n1,patch_frames,n2);
    opts.lambda = 1/sqrt(max(n1,n2)*patch_frames);
    opts.mu = 1e-4;
    opts.tol = 1e-8;
    opts.rho = 1.2;
    opts.max_iter = 800;
    opts.DEBUG = 0;   
    [ L,E,rank] = dictionary_learning( temp, opts);
    
    
    %% approximate L, since sometimes R-TPCA cannot produce a good dictionary
    tho=50;
    Debug = 0;
    [ L_hat,trank,U,V,S ] = prox_low_rank(L,tho);
    if Debug
        fprintf('\n\n ||L_hat-L||=%.3e,  rank=%d\n\n',tnorm(L_hat-L,'fro'),trank);
    end
    LL=tprod(U,S);%%tinny-tsvd
    %% test WSNM-MSL
    max_iter=200;
    TT=C*sqrt(n1*n2);
   
    [Z,tenE,Z_rank,err_va ] = MSLSTIPTv1(temp,LL,TT,p,max_iter,Debug);    
    tenL=tprod(LL,Z);

%% Reconstruct
E_hat=zeros(m*n,onesize);
A_hat=zeros(m*n,onesize);
tenE=reshape(tenE,n1,n2,patch_frames);
tenL=reshape(tenL,n1,n2,patch_frames);
 for i=1:num_image 
      tempEtensor=tenE(:,:,(i-1)*onesize+1:onesize*i);
      tempLtensor=tenL(:,:,(i-1)*onesize+1:onesize*i); 
      for j=1:onesize
          E_hat(:,j)=reshape(tempEtensor(:,:,j),m*n,1);
          A_hat(:,j)=reshape(tempLtensor(:,:,j),m*n,1);
      end
tarImg=reverse_patch_model(E_hat,m,n,mm,nn,endRow,endColumn);  % target image
backImg=reverse_patch_model(A_hat,m,n,mm,nn,endRow,endColumn);
%% threshold processing
temp1=reshape(tarImg,mm*nn,1);
tar_mean=mean(temp1);tar_sigam=std(temp1);
Thre=max(max(temp1)*0.7,0.5*tar_sigam+tar_mean);
index=tarImg>Thre;
tarImg=tarImg.*index;
%% show  
        a=uint8(tarImg*255);  
        figure;
        imshow(a, []);
end 
end
toc