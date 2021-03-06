function facemix(a, b, c, d)
% a: batch b: number
% c: batch d: number
% 1<= a,c <=100
% 1<= b,c <= 16
im = uint8(ones(72, 72)*255);

load trainbatch
load testbatch
load deepauto/mnist_weights

data1 = [batchdata(a,:,b) 1];
[tar1 target1]=max(batchtargets(a,:,b),[],2);
img1 = reshape(data1(1:1024), [32,32])*255;
% target1 = batchtargets(50,:,3);
img1 = (uint8(img1));
w1probs1 = 1./(1 + exp(-data1*w1)); w1probs1 = [w1probs1  1];
w2probs1 = 1./(1 + exp(-w1probs1*w2)); w2probs1 = [w2probs1 1];
w3probs1 = 1./(1 + exp(-w2probs1*w3)); w3probs1 = [w3probs1  1];
w4probs1 = w3probs1*w4; w4probs1 = [w4probs1  1];
w5probs1 = 1./(1 + exp(-w4probs1*w5)); w5probs1 = [w5probs1  1];
w6probs1 = 1./(1 + exp(-w5probs1*w6)); w6probs1 = [w6probs1  1];
w7probs1 = 1./(1 + exp(-w6probs1*w7)); w7probs1 = [w7probs1  1];
dataout1 = 1./(1 + exp(-w7probs1*w8));

% batchdata(1,:,1)
data2 = [batchdata(c,:,d) 1];
% target2 = batchtargets(5,:,1);
[tar2 target2]=max(batchtargets(c,:,d),[],2);
img2 = reshape(data2(1:1024), [32,32])*255;
img2 = (uint8(img2));
w1probs2 = 1./(1 + exp(-data2*w1)); w1probs2 = [w1probs2  1];
w2probs2 = 1./(1 + exp(-w1probs2*w2)); w2probs2 = [w2probs2 1];
w3probs2 = 1./(1 + exp(-w2probs2*w3)); w3probs2 = [w3probs2  1];
w4probs2 = w3probs2*w4; w4probs2 = [w4probs2  1];
w5probs2 = 1./(1 + exp(-w4probs2*w5)); w5probs2 = [w5probs2  1];
w6probs2 = 1./(1 + exp(-w5probs2*w6)); w6probs2 = [w6probs2  1];
w7probs2 = 1./(1 + exp(-w6probs2*w7)); w7probs2 = [w7probs2  1];
dataout2 = 1./(1 + exp(-w7probs2*w8));

img_mixed = (img1 + img2)/2;
img_mixed = (uint8(img_mixed));

%w4probs3 = (w4probs1 + w4probs2)/2;
w3probs3 = (w3probs1 + w3probs2)/2;
w4probs3 = w3probs3*w4; w4probs3 = [w4probs3  1];
w5probs3 = 1./(1 + exp(-w4probs3*w5)); w5probs3 = [w5probs3  1];
w6probs3 = 1./(1 + exp(-w5probs3*w6)); w6probs3 = [w6probs3  1];
w7probs3 = 1./(1 + exp(-w6probs3*w7)); w7probs3 = [w7probs3  1];
dataout3 = 1./(1 + exp(-w7probs3*w8));
img3 = reshape(dataout3, [32,32])*255;
img3 = (uint8(img3));

im(1:32,1:32) = img1;
im(1: 32, 41:72) = img2;
im(41:72, 1:32) = img_mixed;
im(41:72, 41:72) = img3;

str = strcat('Mixed Person ', num2str(target1), 'and ', num2str(target2));
figure('name', str), imshow(im);
end