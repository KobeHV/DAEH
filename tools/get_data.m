function exp_data = get_data(db_name)
 %% Load dataset
fprintf('Loading Data...\n');
if strcmp(db_name, 'MNIST')
    load('mnist_split.mat')
    
    exp_data.WTT = bsxfun(@eq, traingnd, testgnd');
    exp_data.X = [double(traindata) ; double(testdata)];
    exp_data.indexTrain=1 : size(double(traindata),1);
    exp_data.indexTest=size(double(traindata),1)+1 : size(double(traindata),1) + size(double(testdata),1);
    exp_data.label = double([traingnd ; testgnd]);
elseif strcmp(db_name, 'CIFAR-10')
    load('cifar10.mat')

    exp_data.WTT = bsxfun(@eq, train_label_cifar, test_label_cifar');
    exp_data.X = [double(train_data_cifar') ; double(test_data_cifar')];
    exp_data.indexTrain=1 : size(double(train_data_cifar'),1) ;
    exp_data.indexTest=size(double(train_data_cifar'),1) +1 : size(double(train_data_cifar'),1) + size(double(test_data_cifar'),1);
    exp_data.label = double([train_label_cifar ; test_label_cifar]);
elseif strcmp(db_name, 'Caltech-256')
    load('Caltech256_test1000.mat')

    exp_data.WTT = bsxfun(@eq, train_label, test_label');
    exp_data.X = [double(train_data) ; double(test_data)];
    exp_data.indexTrain=1 : size(double(train_data),1);
    exp_data.indexTest=size(double(train_data),1)+1 : size(double(train_data),1) + size(double(test_data),1);
    exp_data.label = double([train_label ; test_label]);
elseif strcmp(db_name, 'ImageNet')
    load('ImageNet128K.mat')

    exp_data.WTT = bsxfun(@eq, traingnd, testgnd');
    exp_data.X = [double(traindata); double(testdata)];
    exp_data.indexTrain=1:size(double(traindata),1);
    exp_data.indexTest=size(double(traindata),1)+1:size(double(traindata),1) + size(double(testdata),1);
    exp_data.label = double([traingnd ; testgnd]);
end