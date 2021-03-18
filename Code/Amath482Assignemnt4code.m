close all; clear all; clc
%%
[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[imagesTest, labelsTest] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');



%%

images = im2double(images);
[m,n,k] = size(images);



%%
for i = 1:k
    mat_image(:,i) = reshape(images(:,:,i),m*n,1);
end

imagesTest = im2double(imagesTest);
[m,n,k] = size(imagesTest);

for i = 1:k
    mat_imageTest(:,i) = reshape(imagesTest(:,:,i),m*n,1);
end

mat_image = im2double(mat_image);
mat_imageTest = im2double(mat_imageTest);

mat0 = mat_image(:,labels == 0);
mat1 = mat_image(:,labels == 1);
mat2 = mat_image(:,labels == 2);
mat3 = mat_image(:,labels == 3);
mat4 = mat_image(:,labels == 4);
mat5 = mat_image(:,labels == 5);
mat6 = mat_image(:,labels == 6);
mat7 = mat_image(:,labels == 7);
mat8 = mat_image(:,labels == 8);
mat9 = mat_image(:,labels == 9);

mean_val = mean(mat_image,2);
[m,n] = size(mat_image);

X = mat_image - repmat(mean_val, 1, n);



%%
[U,S,V] = svd(X/sqrt(n-1),'econ');
u = U;
s = S;
v = V;

[Um,Sm,Vm] = svd(mat_image,'econ');
um = Um;
sm = Sm;
vm = Vm;


%%
plot((diag(S).^2/sum(diag(S).^2)*100), 'ob')
set(gca,'Fontsize',18)
title('Single Values')
xlabel('Index of Single Value')
ylabel('Energy of Single Value (%)')


%%
% must run all section up to this point to work
diagS = diag(S).^2/sum(diag(S).^2)*100;
percent = 0;
i = 1;
while percent < 95
    percent = percent + diagS(i);
    i = i+1;
end
feature = i-1;


%%
proj = U' * X;

scatter3(proj(1,labels==2), proj(2,labels==2), proj(3,labels==2), 'ko');
hold on
scatter3(proj(1,labels==3), proj(2,labels==3), proj(3,labels==3), 'ro');
hold on
scatter3(proj(1,labels==5), proj(2,labels==5), proj(3,labels==5), 'bo');

set(gca,'Fontsize',18)
title('Modes in 3D Space (units insignificant)')
xlabel('Space')
ylabel('Space')
zlabel('Space')

%%
proj = U' * X;

for i = 1:9
    scatter3(proj(2,labels==i), proj(3,labels==i), proj(5,labels==i), 'o');
    hold on
end 

set(gca,'Fontsize',18)
title('3 Modes in Space')
xlabel('First Mode')
ylabel('Second Mode')
zlabel('Third Mode')

%%
proj = U' * X;
scatter3(proj(1,labels==7), proj(2,labels==7), proj(3,labels==7), 'ko');
hold on
scatter3(proj(1,labels==3), proj(2,labels==3), proj(3,labels==3), 'ro');
hold on



%% Calculate scatter matrices

[U,S,V,threshold,w,sort1,sort2] = dc_trainer(mat1,mat2,feature);

mat1Test = mat_imageTest(:,labelsTest == 1);
mat2Test = mat_imageTest(:,labelsTest == 2);
TestSet = [mat1Test mat2Test];

TestNum = size(TestSet,2);
TestMat = U'*TestSet; % PCA 
pval = w'*TestMat;

ResVec = (pval>threshold);

a = zeros(1,size(mat1Test, 2));
b = ones(1, size(mat2Test, 2));

hiddenlabels = [a b];

err = abs(ResVec - hiddenlabels);
err = err > 0;

errNum = sum(err);
sucRate1 = 1 - errNum/TestNum;


%%
%           LDA 10 digits

projTrain = u(:,1:154)' * X;

W = mat_imageTest - repmat(mean_val, 1, size(mat_imageTest,2));
projTest = u(:,1:154)' * W;
projTrain = projTrain/ max(s(:));
projTest = projTest/ max(s(:));

tic
Mdl = fitcdiscr(projTrain(:,1:60000)',labels(1:60000,:)', 'discrimType', 'linear'); 
test_label = predict(Mdl,projTest');
toc

TestNum = size(test_label,1);
err = abs(test_label - labelsTest);
err = err > 0;

errNum = sum(err);
sucRate2 = 1 - errNum/TestNum;
figure(1);
cm = confusionchart(labelsTest,test_label);
title('LDA Classification Confusion')
set(gca,'Fontsize',18)


%%
%           SVM classifier with training data, labels and test set 

projTrain = u(:,1:154)' * X;
W = mat_imageTest - repmat(mean_val, 1, size(mat_imageTest,2));
projTest = u(:,1:154)' * W;
projTrain = projTrain/ max(s(:));
projTest = projTest/ max(s(:));

tic
Mdl = fitcecoc(projTrain(:,1:10000)',labels(1:10000,:)'); 
test_label = predict(Mdl,projTest');
toc

TestNum = size(test_label,1);
err = abs(test_label - labelsTest);
err = err > 0;

errNum = sum(err);
sucRate3 = 1 - errNum/TestNum;
figure(2);
cm = confusionchart(labelsTest,test_label);
title('SVM Classification Confusion')
set(gca,'Fontsize',18)

%%

%           classification tree on fisheriris data 

projTrain = u(:,1:154)' * X;
W = mat_imageTest - repmat(mean_val, 1, size(mat_imageTest,2));
projTest = u(:,1:154)' * W;
projTrain = projTrain/ max(s(:));
projTest = projTest/ max(s(:));

tic
Mdl = fitctree(projTrain',labels','MaxNumSplits',10); 
%view(Mdl ,'Mode','graph'); 
test_label = predict(Mdl,projTest');
toc


TestNum = size(test_label,1);
err = abs(test_label - labelsTest);
err = err > 0;

errNum = sum(err);
sucRate4 = 1 - errNum/TestNum;
figure(3);
cm = confusionchart(labelsTest,test_label);
title('CT Classification Confusion')
set(gca,'Fontsize',18)






















