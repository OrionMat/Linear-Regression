clear;
clc;
close all;

data = loadData('investment.txt');

x = data(:,1:end-1);
y = data(:,end);
m = length(y);
n = size(x, 2);

% add bias feature
X = [ones(m, 1) x]; 

% normal equation
thetaNorm = normalEquation (X, y);
costNorm = costMSE(X, thetaNorm, y, m)

% gradient descent
% feature scaling
scale = (x-mean(x))./std(x);
Xscale = [ones(m, 1) scale];
itterations = 100; 
thetaGrad = zeros(n+1, 1);
alpha = 0.001;
costGrad0 = costMSE(Xscale, thetaGrad, y, m)
[thetaGrad, Jpast] = gradDescent(Xscale, y, thetaGrad, alpha, m, itterations);
costGrad = costMSE(Xscale, thetaGrad, y, m)

% gradient descent with different parameters
thetaA = zeros(n+1, 1);
alphaA = 0.03;
cost0A = costMSE(Xscale, thetaA, y, m)
[thetaA, JpastA] = gradDescent(Xscale, y, thetaA, alphaA, m, itterations);
costA = costMSE(Xscale, thetaA, y, m)

thetaB = zeros(n+1, 1);
alphaB = 0.1;
cost0B = costMSE(Xscale, thetaB, y, m)
[thetaB, JpastB] = gradDescent(Xscale, y, thetaB, alphaB, m, itterations);
costB = costMSE(Xscale, thetaB, y, m)

thetaC = zeros(n+1, 1);
alphaC = 0.3;
cost0C = costMSE(Xscale, thetaC, y, m)
[thetaC, JpastC] = gradDescent(Xscale, y, thetaC, alphaC, m, itterations);
costC = costMSE(Xscale, thetaC, y, m)

thetaD = zeros(n+1, 1);
alphaD = 1;
cost0D = costMSE(Xscale, thetaD, y, m)
[thetaD, JpastD] = gradDescent(Xscale, y, thetaD, alphaD, m, itterations);
costD = costMSE(Xscale, thetaD, y, m)



% plotting 

figure 1;

plot(0:1:itterations, [costGrad0 Jpast], 'k');
hold on
plot(0:1:itterations, [cost0A JpastA], 'r');
plot(0:1:itterations, [cost0B JpastB], 'm');
plot(0:1:itterations, [cost0C JpastC], 'b');
plot(0:1:itterations, [cost0C JpastD], 'g');
title('Gradient Descent Convergence');
legend('alpha 0.03', 'alpha 0.1', 'alpha 0.3', 'alpha 1');


