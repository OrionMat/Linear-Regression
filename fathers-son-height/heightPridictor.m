clear;
clc;
close all;

data = loadData('heights.txt');

x = data(:,1:end-1);
y = data(:,end);
m = length(y);

% add bias feature
X = [ones(m, 1) x]; 

% normal equation
thetaNorm = normalEquation (X, y);
costNorm = costMSE(X, thetaNorm, y, m)

% gradient descent
thetaGrad = [0 ; 0];
alpha = 0.0001;
itterations = 25;
costGrad0 = costMSE(X, thetaGrad, y, m)
[thetaGrad, Jpast] = gradDescent(X, y, thetaGrad, alpha, m, itterations);
costGrad = costMSE(X, thetaGrad, y, m)

% feature scaling and gradient descent
scale = (x-mean(x))./std(x);
Xscale = [ones(m, 1) scale]; 
thetaGradScale = [0 ; 0];
alphaScale = 1;
costGrad0Scale = costMSE(Xscale, thetaGradScale, y, m)
[thetaGradScale, JpastScale] = gradDescent(Xscale, y, thetaGradScale, alphaScale, m, itterations);
costGradScale = costMSE(Xscale, thetaGradScale, y, m)





 
% gradient descent with different parameters
thetaA = [0 ; 0];
alphaA = 0.3;
cost0A = costMSE(Xscale, thetaA, y, m)
[thetaA, JpastA] = gradDescent(Xscale, y, thetaA, alphaA, m, itterations);
costA = costMSE(Xscale, thetaA, y, m)

thetaB = [0 ; 0];
alphaB = 0.03;
cost0B = costMSE(Xscale, thetaB, y, m)
[thetaB, JpastB] = gradDescent(Xscale, y, thetaB, alphaB, m, itterations);
costB = costMSE(Xscale, thetaB, y, m)

thetaC = [0 ; 0];
alphaC = 0.01;
cost0C = costMSE(Xscale, thetaC, y, m)
[thetaC, JpastC] = gradDescent(Xscale, y, thetaC, alphaC, m, itterations);
costB = costMSE(Xscale, thetaC, y, m)




 
% plotting 

theta0 = linspace(-20, 80, 100)';
theta1 = linspace(-15, 15, 100)';
J_vals = costValues(X, y, theta0, theta1, m);

theta0scale = linspace(-5, 140, 100)';
theta1scale = linspace(-100, 100, 100)';
J_valScale = costValues(Xscale, y, theta0scale, theta1scale, m);

figure 1;
% figure 1 subplot 1: raw data
subplot(2,2,1); 
plot(x, y, 'rx');
xlabel('fathers height (inches)');
ylabel('sons height (inches)');
title('Father-Son Heights');

% figure 1 subplot 2: cost function
subplot(2,2,2);
surf(theta0, theta1, J_vals);
xlabel('\theta_0'); 
ylabel('\theta_1');
zlabel('Cost');
title('MSE Cost function');

% figure 1 subplot 3: hypothesis
hxNorm = X*thetaNorm;
hxGrad = X*thetaGrad;
subplot(2,2,3);
plot(x, y, 'rx');
hold on;
plot(x, hxNorm, 'b'); % normal equation
plot(x, hxGrad, 'g'); % gradient descent
xlabel('fathers height (inches)');
ylabel('sons height (inches)');
title('Father-Son Predictors');

% figure 1 subplot 4: contour plot
subplot(2,2,4);
contour(theta0, theta1, J_vals, logspace(-8, 10, 20))
hold on;
plot(thetaNorm(1), thetaNorm(2), 'bx', 'MarkerSize', 10, 'LineWidth', 2);  % normal theta values
plot(thetaGrad(1), thetaGrad(2), 'gx', 'MarkerSize', 10, 'LineWidth', 2); % grad descent theta values
xlabel('\theta_0'); 
ylabel('\theta_1');
title('MSE Cost Contours');

figure 2;
% figure 2 subplot 1: scaled cost function
subplot(2,2,1);
surf(theta0scale, theta1scale, J_valScale);
xlabel('\theta_0 scaled'); 
ylabel('\theta_1 scaled');
zlabel('Cost');
title('scaled MSE Cost');

% figure 2 subplot 2: convergence rate
subplot(2,2,2);
plot(0:1:itterations, [costGrad0 Jpast], 'g');
hold on
plot(0:1:itterations, [costGrad0Scale JpastScale], 'r');
plot(0:1:itterations, [cost0A JpastA], 'm');
plot(0:1:itterations, [cost0B JpastB], 'k');
plot(0:1:itterations, [cost0C JpastC], 'b');
title('Gradient Descent Convergence');

% figure 2 subplot 3: contour plot
subplot(2,2,3);
contour(theta0scale, theta1scale, J_valScale, logspace(-8, 10, 20))
hold on;
plot(thetaGradScale(1), thetaGradScale(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2); % grad descent theta values
plot(thetaA(1), thetaA(2), 'mx', 'MarkerSize', 10, 'LineWidth', 2); % grad descent theta values
plot(thetaB(1), thetaB(2), 'kx', 'MarkerSize', 10, 'LineWidth', 2); % grad descent theta values
plot(thetaC(1), thetaC(2), 'bx', 'MarkerSize', 10, 'LineWidth', 2); % grad descent theta values
xlabel('\theta_0'); 
ylabel('\theta_1');
title('Scaled MSE Cost Contours');


