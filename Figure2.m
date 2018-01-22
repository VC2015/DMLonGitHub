rng(13867)

B = 10000;
n = 500;

b1 = zeros(B,1);
b2 = zeros(B,1);

for ii = 1:B
    zh = randn(n/2,1);
    z = [zh;zh];
    u = randn(n,1);
    v = randn(n,1);
    x = z + u;
    y = x + z + v;
    
    mhat = z;
    overfit = (y-z)/(n^(1/3));
    
    b1(ii,1) = ((x-mhat)'*(y-z-overfit))/((x-mhat)'*x);
    
    splitA = (1:(n/2))';
    splitB = ((n/2)+1:n)';
    
    bA = (u(splitA)'*(y(splitA)-z(splitA)-overfit(splitB)))/(u(splitA)'*x(splitA));
    bB = (u(splitB)'*(y(splitB)-z(splitB)-overfit(splitA)))/(u(splitB)'*x(splitB));
    b2(ii,1) = (1/2)*(bA+bB);
end

edges = -7:.05:7;
[nb1,xb1] = histcounts((b1-1)/std(b1),edges,'Normalization','pdf');
[nb2,xb2] = histcounts((b2-1)/std(b2),edges,'Normalization','pdf');

figure;
subplot(1,2,1), bar((xb1(1:end-1)+xb1(2:end))/2,nb1,1,'FaceColor',[0,.5,.75],'FaceAlpha',.5);
hold on;
plot(edges,normpdf(edges),'r');
title('A.  Full Sample')
legend('Simulation','N(0,1)','Location','NorthWest')
subplot(1,2,2), bar((xb2(1:end-1)+xb2(2:end))/2,nb2,1,'FaceColor',[0,.5,.75],'FaceAlpha',.5);
hold on;
plot(edges,normpdf(edges),'r');
title('B.  Split Sample')
legend('Simulation','N(0,1)','Location','NorthWest')



    