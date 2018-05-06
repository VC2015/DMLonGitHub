function [se,viid] = iid_se(x,e,XpXinv)

k = size(x,2);
n = size(x,1);

viid = (e'*e/(n-k))*XpXinv;

se = sqrt(diag(viid));

