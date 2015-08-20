load deep_classify\mnistclassify_weights.mat
load testbatch

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
  index = find(J~=J1);
  img = reshape(data(index,1:1024), [32,32])*255;
  str = strcat('ture: ',num2str(J1(index)),'Predicted: ',num2str(J(index)));
  figure('name', str), imshow(uint8(img));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
end
 test_err=(testnumcases*testnumbatches-counter);
 test_crerr=err_cr/testnumbatches;