clear;
clc;

global KTYPE
global KSCALE
global online
global visualize
visualize = 0;
KTYPE = 1;
KSCALE = 0.25;
online = 1;
C = 1;

file = 'C://Users/gonza647/CS/Courses/CSE847-Machine Learning/Project/Code/data.txt';
%fileID = fopen(file, 'r');
%M = dlmread(file,',');
[a b c d labels] = textread(file, '%f %f %f %f %s', -1, 'delimiter', ',');
data = [a b c d]; 
samples = length(data);
xtest = zeros(1,4); xtrain = zeros(1,4);
ytest = 0;          ytrain = 0;
[trainInd, valInd, testInd] = dividerand(samples,0.7,0,0.3);    % Randomly sample indexes 
for i =1:samples                                                % Split data into training and testing sets
   if ismember(i,trainInd)
       xtrain = [xtrain;data(i,:)];
       if strcmp(labels(i),'Iris-versicolor')
           ytrain = [ytrain;1];
       else 
           ytrain = [ytrain;-1];
       end
   else 
       xtest = [xtest;data(i,:)];
       if strcmp(labels(i),'Iris-versicolor')
           ytest = [ytest;1];
       else 
           ytest = [ytest;-1];
       end
   end
end
xtrain = xtrain(2:end,:); xtest = xtest(2:end,:);
ytrain = ytrain(2:end,:); ytest = ytest(2:end,:);

[a,b,D,inds,inde,indwLOO] = svcm_train(xtrain,ytrain,C);
[ypred,indwTEST] = svcm_test(xtest,ytest,xtrain,ytrain,a,b);
