function [ L,E,rank] = dictionary_learning( XX, opts)
[L,E,rank,obj,err,iter] = trpca_tnn(XX,opts.lambda,opts);
end

