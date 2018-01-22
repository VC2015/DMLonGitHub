clear;

try %#ok<TRYNC>
    parpool();
end

nrep = 5000;

n = 500;
p = 20;

S = chol(toeplitz(.7.^(0:p-1)));

rng(2162016);

rfsss1 = zeros(nrep,2);
rfgsss1 = zeros(nrep,2);
rfsds1 = zeros(nrep,2);
rfsss2 = zeros(nrep,2);
rfgsss2 = zeros(nrep,2);
rfsds2 = zeros(nrep,2);
rfsss = zeros(nrep,2);
rfgsss = zeros(nrep,2);
rfsds = zeros(nrep,2);

for ii = 1:nrep
    x = randn(n,p)*S;
    
    a0 = 1;
    a1 = .25;
    s1 = 1;
    b0 = 1;
    b1 = .25;
    s2 = 1;
    alpha = .5;
    d = a0*x(:,1)+a1*exp(x(:,3))./(1+exp(x(:,3)))+s1*randn(n,1);
    y = alpha*d+b0*exp(x(:,1))./(1+exp(x(:,1)))+b1*x(:,3)+s2*randn(n,1);
        
    % Sample splitting
    samp = randsample(n,floor(n/2));
    osamp = setdiff(1:n,samp);
    
    % Use first split
    fy1 = TreeBagger(500,x(samp,:),y(samp,:),'Method','regression','Options',statset('UseParallel',true));
    yhat1 = predict(fy1,x(osamp,:));
    ry1 = y(osamp,:) - yhat1;
    
    fd1 = TreeBagger(500,x(samp,:),d(samp,:),'Method','regression','Options',statset('UseParallel',true));
    dhat1 = predict(fd1,x(osamp,:));
    rd1 = d(osamp,:) - dhat1;
    
    rfsss1(ii,1) = (rd1'*d(osamp,:))\(rd1'*y(osamp,:));
    rfsss1(ii,2) = sqrt(mean((y(osamp,:)-d(osamp,:)*rfsss1(ii,1)).^2)*(rd1'*rd1)/((rd1'*d(osamp,:))^2));

    ghat1 = yhat1 - dhat1*rfsss1(ii,1);
    gy1 = y(osamp,:)-ghat1;
    rfgsss1(ii,1) = d(osamp,:)\gy1;
    rfgsss1(ii,2) = iid_se(d(osamp,:),gy1-d(osamp,:)*rfgsss1(ii,1),inv(d(osamp,:)'*d(osamp,:)));

    rfsds1(ii,1) = rd1\ry1;
    rfsds1(ii,2) = iid_se(rd1,ry1-rd1*rfsds1(ii,1),inv(rd1'*rd1));
    
    % Use second split
    fy2 = TreeBagger(500,x(osamp,:),y(osamp,:),'Method','regression','Options',statset('UseParallel',true));
    yhat2 = predict(fy2,x(samp,:));
    ry2 = y(samp,:) - yhat2;
    
    fd2 = TreeBagger(500,x(osamp,:),d(osamp,:),'Method','regression','Options',statset('UseParallel',true));
    dhat2 = predict(fd2,x(samp,:));
    rd2 = d(samp,:) - dhat2;
    
    rfsss2(ii,1) = (rd2'*d(samp,:))\(rd2'*y(samp,:));
    rfsss2(ii,2) = sqrt(mean((y(samp,:)-d(samp,:)*rfsss2(ii,1)).^2)*(rd2'*rd2)/((rd2'*d(samp,:))^2));

    ghat2 = yhat2 - dhat2*rfsss2(ii,1);
    gy2 = y(samp,:)-ghat2;
    rfgsss2(ii,1) = d(samp,:)\gy2;
    rfgsss2(ii,2) = iid_se(d(samp,:),gy2-d(samp,:)*rfgsss2(ii,1),inv(d(samp,:)'*d(samp,:)));

    rfsds2(ii,1) = rd2\ry2;
    rfsds2(ii,2) = iid_se(rd2,ry2-rd2*rfsds2(ii,1),inv(rd2'*rd2));    
    
    % Average
    rfsss(ii,1) = .5*rfsss1(ii,1)+.5*rfsss2(ii,1);
    rfsss(ii,2) = sqrt(.25*rfsss1(ii,2)^2+.25*rfsss2(ii,2)^2);
    
    rfgsss(ii,1) = .5*rfgsss1(ii,1)+.5*rfgsss2(ii,1);
    rfgsss(ii,2) = sqrt(.25*rfgsss1(ii,2)^2+.25*rfgsss2(ii,2)^2);

    rfsds(ii,1) = .5*rfsds1(ii,1)+.5*rfsds2(ii,1);
    rfsds(ii,2) = sqrt(.25*rfsds1(ii,2)^2+.25*rfsds2(ii,2)^2);
        
    if ii/10 == floor(ii/10)
        disp(ii)
        disp(mean(abs(rfsds(1:ii,1) - alpha)./rfsds(1:ii,2) > 1.96))
        disp(mean(abs(rfsss(1:ii,1) - alpha)./rfsss(1:ii,2) > 1.96))
        disp(mean(abs(rfgsss(1:ii,1) - alpha)./rfgsss(1:ii,2) > 1.96))
    end
    
end
disp('RF - Nonorthogonal (D) + sample splitting');   
disp(mean(abs(rfsss(:,1) - alpha)./rfsss(:,2) > 1.96))
disp('RF - Nonorthogonal (Y) + sample splitting');   
disp(mean(abs(rfgsss(:,1) - alpha)./rfgsss(:,2) > 1.96))
disp('RF - Orthogonal + sample splitting');
disp(mean(abs(rfsds(:,1) - alpha)./rfsds(:,2) > 1.96))


% Overlaid histograms. Centered and rescaled by simulation standard error.  
% Random forest results
se1 = std(rfgsss(:,1));
se2 = std(rfsds(:,1));
sc = max([se1,se2]);

tse1 = se1*(rfgsss(:,1) - alpha)./rfgsss(:,2);
tse2 = se2*(rfsds(:,1) - alpha)./rfsds(:,2);

lb = min([min(tse1),min(tse2)])-.25*sc;
ub = max([max(tse1),max(tse2)])+.25*sc;
lb = -max(abs(lb),abs(ub));
ub = max(abs(lb),abs(ub));

[nss,xss] = hist(tse1,floor(sqrt(nrep)));
pss = (nss/sum(nss))/(xss(2)-xss(1));
[nds,xds] = hist(tse2,floor(sqrt(nrep)));
pds = (nds/sum(nds))/(xds(2)-xds(1));

ym = max([max(pss),max(pds)]);
ym = 1.05*max([max(normpdf(lb:.001:ub,0,se1)),max(normpdf(lb:.001:ub,0,se2)),ym]);



figure; 
bar(xss,pss,1,'FaceColor',[0,.5,.75],'FaceAlpha',.5);
axis([lb,ub,0,ym]);
hold on;
plot(lb:.001:ub,normpdf(lb:.001:ub,0,se1),'r')
ti = sprintf('Non-Orthogonal, n = %d, p = %d',n,p);
title(ti)
legend('Simulation','N(0,\Sigma_s)','Location','NorthWest')

figure; 
bar(xds,pds,1,'FaceColor',[0,.5,.75],'FaceAlpha',.5);
axis([lb,ub,0,ym]);
hold on;
plot(lb:.001:ub,normpdf(lb:.001:ub,0,se2),'r')
ti = sprintf('Orthogonal, n = %d, p = %d',n,p);
title(ti)
legend('Simulation','N(0,\Sigma_s)','Location','NorthWest')

% Overlaid histograms
figure;
edges = .40:.015:.71;
[ngsss,xgsss] = histcounts(rfgsss(:,1),edges,'Normalization','pdf');
[nsss,xsss] = histcounts(rfsss(:,1),edges,'Normalization','pdf');
[nsds,xsds] = histcounts(rfsds(:,1),edges,'Normalization','pdf');

bar((xgsss(1:end-1)+xgsss(2:end))/2,ngsss,1,'FaceColor',[0,0,1],'FaceAlpha',.25);
hold on;
bar((xsds(1:end-1)+xsds(2:end))/2,nsds,1,'FaceColor',[1,0,0],'FaceAlpha',.25);
plot([.5;.5],[0;12],'k');
legend('Non-Orthogonal','Orthogonal');
title('Simulated Distributions of $\widehat\theta$','interpreter','latex');


