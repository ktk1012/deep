% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program fine-tunes an autoencoder with backpropagation.
% Weights of the autoencoder are going to be saved in mnist_weights.mat
% and trainig and test reconstruction errors in mnist_error.mat
% You can also set maxepoch, default value is 200 as in our paper.  

maxepoch=200;
fprintf(1,'\nTraining discriminative model on MNIST by minimizing cross entropy error. \n');
fprintf(1,'60 batches of 1000 cases each. \n');

load deep_classify/mnistvhclassify
load deep_classify/mnisthpclassify
% load deep_classify/mnisthp2classify
% load deep_classify/mnisthp3classify

%makebatches;

load trainbatch
load testbatch

[numcases numdims numbatches]=size(batchdata);
N=numcases; 

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
% w3=[hidpen2; penrecbiases2];
% w4=[hidpen3; penrecbiases3];
w_class = 0.1*randn(size(w2,2)+1,10);
 

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;
l2=size(w2,1)-1;
% l3=size(w3,1)-1;
% l4=size(w4,1)-1;
% l5=size(w_class,1)-1;
% l6=10; 
l3=size(w_class,1)-1;
l4=10; 
test_err=[];
train_err=[];


for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0; 
err_cr=0;
counter=0;
[numcases numdims numbatches]=size(batchdata);
N=numcases;
 for batch = 1:numbatches
  data = [batchdata(:,:,batch)];
  target = [batchtargets(:,:,batch)];
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
%   w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
%   w4probs = 1./(1 + exp(-w3probs*w4)); w4probs = [w4probs  ones(N,1)];
  targetout = exp(w2probs*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,10);

  [I J]=max(targetout,[],2);
  [I1 J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
 end
 train_err(epoch)=(numcases*numbatches-counter);
 train_crerr(epoch)=err_cr/numbatches;

%%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0;
err_cr=0;
counter=0;
[testnumcases testnumdims testnumbatches]=size(testbatchdata);
N=testnumcases;
for batch = 1:testnumbatches
  data = [testbatchdata(:,:,batch)];
  target = [testbatchtargets(:,:,batch)];
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
%   w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
%   w4probs = 1./(1 + exp(-w3probs*w4)); w4probs = [w4probs  ones(N,1)];
  targetout = exp(w2probs*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,10);

  [I J]=max(targetout,[],2); % return maximum value and the index of maximum value
  [I1 J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
end
 test_err(epoch)=(testnumcases*testnumbatches-counter);
 test_crerr(epoch)=err_cr/testnumbatches;
 fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
            epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);

%%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
 for batch = 1:numbatches
 fprintf(1,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 data=batchdata(:,:,batch);
 targets=batchtargets(:,:,batch);
 
%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  max_iter=3;

  if epoch<21  % First update top-level weights holding other weights fixed. 
    N = size(data,1);
    XX = [data ones(N,1)];
    w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2)); %w2probs = [w2probs ones(N,1)];
%     w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
%     w4probs = 1./(1 + exp(-w3probs*w4)); %w4probs = [w4probs  ones(N,1)];
    
    VV = [w_class(:)']';
    Dim = [l3; l4];
    [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',max_iter,Dim,w2probs,targets);
    w_class = reshape(X,l3+1,l4);

  else
    VV = [w1(:)' w2(:)' w_class(:)']';
    Dim = [l1; l2; l3; l4];
    [X, fX] = minimize(VV,'CG_CLASSIFY',max_iter,Dim,data,targets);

    w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
    xxx = (l1+1)*l2;
    w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
    xxx = xxx+(l2+1)*l3;
%     w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
%     xxx = xxx+(l3+1)*l4;
%     w4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
%     xxx = xxx+(l4+1)*l5;
    w_class = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);

  end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end

 save deep_classify/mnistclassify_weights w1 w2 w_class
 save deep_classify/mnistclassify_error test_err test_crerr train_err train_crerr;

end



