function [B_train , B_test , loss ] = DAEH (data_our, opt, nbits)

[N1,d] = size(data_our.X); %Ntrain
X  = data_our.X(data_our.indexTrain, :); X=X'; %train
X2 = data_our.X(data_our.indexTest, :); X2 = X2';%test
y  = data_our.label(data_our.indexTrain, :);

% label matrix Y = N x c   (c is amount of classes, N is amount of instances)
if isvector(y)
    Y = sparse(1:length(y), double(y), 1); Y = full(Y');
else
    Y = y';
end
n_cls = size(Y,1);

% %%%%%% B - initialize %%%%%%
B=randn(nbits,N1)>0; B=B*2-1;
% %%%%%% W - initialize %%%%%%
W1  = randn(nbits,d);
W2  = randn(nbits, size(Y,1)); %W=B*Y';
YYT = Y*Y';
B   = sign(W2*Y);

%-----------------------------------------------------training---------------------------------
loss = zeros(opt.Iter_num,1);
for iter=1:opt.Iter_num
    fprintf('%d...',iter);
    
    Q = W1*X + opt.lambda*W2*Y;
    Q = Q';

    B = zeros(size(B'));  
    Wg = W1;
    for time = 1:opt.max_iter_B          
        Z0 = B;
        for k = 1 : size(B,2)
            Zk = B; Zk(:,k) = [];
            Wkk = Wg(k,:); Wk = Wg; Wk(k,:) = [];                    
            B(:,k) = sign(Q(:,k) -  Zk*Wk*Wkk');
        end
        fprintf('Main Iter: %d, inner iter = %d, value = %.2f...\n', iter, time, norm(B-Z0,'fro'));
        if norm(B-Z0,'fro') < 1e-6 * norm(Z0,'fro')
            break
        end
    end
    B = B';%B = W2*Y;
    W1 = (B*B'+opt.beta*eye(nbits))\(B*X');
    W2 = B*Y'/(YYT+opt.beta*eye(n_cls));
    x1 = norm(W1'*B-X,'fro')
    x2 = norm(W2*Y-B,'fro')
    loss(iter,1) = x1 + opt.lambda*x2
%     loss(iter,1) = x2
end


B_train=B'>0;
%--Out-of-Sample------
NT = (X * X' + 1 * eye(size(X, 1))) \ X;
W  = NT*B_train;
B_test=X2'*W>0;
 
end



