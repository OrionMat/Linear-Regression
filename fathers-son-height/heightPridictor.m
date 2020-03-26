clear;
clc;
close all;

load('-ascii', 'heights.txt');

x = heights(:,1);
y = heights(:,2);
m = length(y);

figure 1;
subplot(2,2, 1); 
plot(x, y, 'rx');
xlabel('fathers height (inches)');
ylabel('sons height (inches)');
title('Father-Son Heights');
hold on

% bias feature
X = [ones(m, 1) x]; 

% cost graph
plotCost(X, y, m)

% normal equation
theta = pinv(X'*X)*X'*y;
hx = X*theta;
costMSE(X, theta, y, m)

figure 1;
subplot(2,2, 2); 
hold on
plot(x, y, 'rx');
plot(x, hx, 'b');
figure 3;
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 20, 'LineWidth', 4)

% gradient descent
thetaGrad = [0 ; 0];
alpha = 0.0001;
itterations = 25;
cost0 = costMSE(X, thetaGrad, y, m)

% feature scaling
%scaled = (heights-mean(heights))./std(heights);
%Xscaled = [ones(m, 1) scaled]; 

[thetaGrad, Jpast] = gradDescent(X, thetaGrad, y, alpha, m, itterations);
costMSE(X, thetaGrad, y, m)
hx = X*thetaGrad;

figure 1;
subplot(2,2, 4); 
plot(x, y, 'rx');
hold on;
plot(x, hx, 'b');
subplot(2,2, 3); 
plot(0:1:itterations, [cost0 Jpast], 'b');

figure 3;
hold on;
plot(thetaGrad(1), thetaGrad(2), 'bx', 'MarkerSize', 20, 'LineWidth', 4);

% gradient descent different initial conditions
thetaGrad = [60 ; 0];
alpha =  0.0001;
itterations = 25;
cost0 = costMSE(X, thetaGrad, y, m)

[thetaGrad, Jpast] = gradDescent(X, thetaGrad, y, alpha, m, itterations);
costMSE(X, thetaGrad, y, m)
hx = X*thetaGrad;

figure 1;
subplot(2,2, 4); 
hold on;
plot(x, hx, 'm');
subplot(2,2, 3); 
hold on;
plot(0:1:itterations, [cost0 Jpast], 'm');

figure 3;
hold on;
plot(thetaGrad(1), thetaGrad(2), 'mx', 'MarkerSize', 20, 'LineWidth', 4);
