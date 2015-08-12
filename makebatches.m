%%% training batch data
FileList = dir('training/*.png');

totnum = length(FileList);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches = totnum/100;
batchsize = 100;
numdims = 32^2;
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, 10, numbatches);

for i=1:numbatches
    for j=1:batchsize
        index = randomorder((i-1)*batchsize+j);
        a = strsplit(FileList(index).name,'_');
        c = str2num(cell2mat(a(1)));
        batchtargets(j,c,i) = 1;
        img = imread(strcat('training/', FileList(index).name));
        batchdata(j,:,i) = reshape(double(img), [1024,1])/255;
    end
end
save trainbatch batchdata batchtargets;

%%%% test batch
FileList = dir('test/*.png');

totnum = length(FileList);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);
numbatches = totnum/100;
batchsize = 100;
numdims = 32^2;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, 10, numbatches);

for i=1:numbatches
    for j=1:batchsize
        index = randomorder((i-1)*batchsize+j);
        a = strsplit(FileList(index).name,'_');
        c = str2num(cell2mat(a(1)));
        testbatchtargets(j,c,i) = 1;
        img = imread(strcat('test/', FileList(index).name));
        testbatchdata(j,:,i) = reshape(double(img), [1024,1])/255;
    end
end

save testbatch testbatchdata testbatchtargets;

clear a c img;

rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 
