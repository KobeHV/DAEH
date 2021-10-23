function [recall, precision, mAP, rec, pre, retrieved_list] = demo(exp_data, param, method)
% input: 
%          data: 
%              data.train_data
%              data.test_data
%              data.db_data
%          param:
%              param.nbits---encoding length
%              param.pos---position
%          method: encoding length
% output:
%            recall: recall rate
%            precision: precision rate
%            evaluation_info: 
% trueRank = exp_data.knn_p2;

WtrueTestTraining = exp_data.WTT;
pos = param.pos;

%% methods
switch(method)
    % DAEH method 
    case 'DAEH'
        fprintf('......%s start...... \n\n', 'DAEH');
        % Anchor features
        train_data = exp_data.X(exp_data.indexTrain , :);
        test_data = exp_data.X(exp_data.indexTest , :);
        train_label = exp_data.label(exp_data.indexTrain , :);
        test_label = exp_data.label(exp_data.indexTest , :);
        X = train_data;
        n_anchors = 1000; 
        anchor = X(randsample( size(train_data,1), n_anchors),:);
        Dis = EuDist2(X, anchor, 0);
        sigma = mean(min(Dis,[],2).^0.5);
        if strcmp(param.db_name,'CIFAR-10')
            sigma = 0.4; 
        end
        Phi_testdata = exp(-sqdist_sdh(test_data,anchor)/(2*sigma*sigma));
        Phi_traindata = exp(-sqdist_sdh(train_data,anchor)/(2*sigma*sigma));
        X = [Phi_traindata ; Phi_testdata];
        data_our.indexTrain = exp_data.indexTrain;
        data_our.indexTest = exp_data.indexTest;
        data_our.X = normZeroMean(X);
        data_our.X = normEqualVariance(X);
%         data_our.label = double([train_label_cifar;test_label_cifar]);        
        data_our.label = double([train_label ; test_label]);
        
        % nbits=nbits_set(ii);    
        % parameters
        opt.lambda      = param.lambda;
        opt.beta           = param.beta
        opt.Iter_num    = 5;
        opt.max_iter_B  = 10;
        
        [B_trn, B_tst, ~] = DAEH(data_our, opt, param.nbits);
        B_trn = compactbit(B_trn);
        B_tst = compactbit(B_tst);
    % HCOH
    case 'HCOH'
        addpath('./compare/HCOH/');
        fprintf('......%s start...... \n\n', 'HCOH');
        [B_trn, B_tst] = HCOH(exp_data, param.nbits);
        B_trn = compactbit(B_trn);
        B_tst = compactbit(B_tst);
    % BSODH
    case 'BSODH'
        addpath('./compare/BSODH/');
        fprintf('......%s start...... \n\n', 'BSODH');
        [B_trn, B_tst] = BSODH(exp_data, param.nbits);    
    case 'ADLLE'
        addpath('./compare/DLLH/');
        fprintf('......%s start...... \n\n', 'DLLH');
        [B_trn, B_tst] = ADLLE_test(exp_data.X,param.nbits,'ADLLE')
    case 'COSDISH'
        addpath('./compare/COSDISH/');
        fprintf('......%s start...... \n\n', 'COSDISH');
        [B_trn, B_tst] = COSDISH(exp_data,param.nbits);
    % ITQ method proposed in CVPR11 paper
    case 'KSH'
        addpath('./compare/KSH/');
        fprintf('......%s start...... \n\n', 'KSH');
        [B_trn, B_tst] = KSH_Func(exp_data,param.nbits);
    case 'SCDH'
        addpath('./compare/SCDH/');
        fprintf('......%s start...... \n\n', 'SCDH');
        [B_trn, B_tst] = run_scdh(exp_data,param.nbits);
    case 'FSDH'
        addpath('./compare/FSDH/');
        fprintf('......%s start...... \n\n', 'FSDH');
        [B_trn, B_tst] = FSDH_test(exp_data,param.nbits);
%         [B_trn, B_tst] = FSDH_old(exp_data,param.nbits);
    case 'HCSDH'
        addpath('./compare/HCSDH/');
        fprintf('......%s start...... \n\n', 'HCSDH');
        [B_trn, B_tst] = HCSDH_test(exp_data,param);
     case 'HMOH'
        addpath('./compare/HMOH/');
        fprintf('......%s start...... \n\n', 'HMOH');
        [B_trn, B_tst] = HMOH(exp_data,param.nbits);
        B_trn = compactbit(B_trn);
        B_tst = compactbit(B_tst);
    case 'FSSH'
        addpath('./compare/FSSH/');
        fprintf('......%s start...... \n\n', 'FSSH');
        [B_trn, B_tst] = FSSH_test(exp_data,param);
    case 'LFH'
        addpath('./compare/LFH/');
        fprintf('......%s start...... \n\n', 'LFH');
        [B_trn, B_tst] = LFH_test(exp_data,param.nbits);
        B_trn = compactbit(B_trn);
        B_tst = compactbit(B_tst);
    case 'SDH'
        addpath('./compare/SDH/');
        fprintf('......%s start...... \n\n', 'SDH');
        [B_trn, B_tst] = SDH_test(exp_data,param.nbits);
%         B_trn = compactbit(B_trn);
%         B_tst = compactbit(B_tst);
    case 'ITQ'
        addpath('./compare/ITQ/');
        fprintf('......%s start...... \n\n', 'ITQ');
        [B_trn, B_tst] = ITQ_test(exp_data,param.nbits,'ITQ');
    % SGH hashing
    case 'SGH'
        addpath('./compare/SGH/');
        fprintf('......%s start...... \n\n', 'SGH');
        [B_trn, B_tst] = test_SGH(exp_data,param.nbits);
    % Locality sensitive hashing (LSH)
     case 'LSH'
        addpath('./compare/Method-LSH/');
        fprintf('......%s start ......\n\n', 'LSH');
        train_data = exp_data.X(exp_data.indexTrain , :);
        test_data = exp_data.X(exp_data.indexTest , :);
        [~, D] = size(train_data);
        LSHparam.nbits = param.nbits;
        LSHparam.dim = D;
        LSHparam = trainLSH(LSHparam);
        [B_trn, ~] = compressLSH(train_data, LSHparam);
        [B_tst, ~] = compressLSH(test_data, LSHparam);
        %[B_db, ~] = compressLSH(db_data, LSHparam);
        clear db_data LSHparam;
end

%% evaluate
B_trn_size = size(B_trn)
B_tst_size = size(B_tst)

% B1 = compactbit(B_trn);
% B2 = compactbit(B_tst);
Dhamm = hammingDist(B_tst, B_trn);
size_Dhamm = size(Dhamm)
% Dhamm = hammingDist(B2, B1);
[~, rank] = sort(Dhamm, 2, 'ascend');

% clear B_tst B_trn;
choice = param.choice;
switch(choice)
    case 'evaluation_PR_MAP'
        clear train_data test_data;
        [recall, precision, ~] = recall_precision(WtrueTestTraining', Dhamm);
        [rec, pre]= recall_precision5(WtrueTestTraining', Dhamm, pos); % recall VS. the number of retrieved sample
%         [mAP] = area_RP(recall, precision);
        [mAP] = calcMAP(rank, WtrueTestTraining');
        fprintf('==> Method: %s, Bits: %d, MAP: %.4f...   \n', method, param.nbits, mAP);
        retrieved_list = [];
    case 'evaluation_PR'
        clear train_data test_data;
        eva_info = eva_ranking(rank, trueRank, pos);
        rec = eva_info.recall;
        pre = eva_info.precision;
        recall = [];
        precision = [];
        mAP = [];
        retrieved_list = [];
%     case 'visualization'
%         num = param.numRetrieval;
%         retrieved_list =  visualization(Dhamm, ID, num, train_data, test_data); 
%         recall = [];
%         precision = [];
%         rec = [];
%         pre = [];
%         mAP = [];
end

end
