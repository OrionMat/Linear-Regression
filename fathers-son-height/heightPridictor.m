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
% feature scaling
scale = (x-mean(x))./std(x);
Xscale = [ones(m, 1) scale]; 

costGrad0 = costMSE(X, thetaGrad, y, m)
[thetaGrad, Jpast] = gradDescent(X, y, thetaGrad, alpha, m, itterations);
costGrad = costMSE(X, thetaGrad, y, m)


% plotting 

theta0 = linspace(-20, 80, 100)';
theta1 = linspace(-3, 3, 100)';
J_vals = costValues(X, y, theta0, theta1, m);

figure 1;
% figure 1 subplot 1: raw data
subplot(2,2,1); 
plot(x, y, 'rx');
xlabel('fathers height (inches)');
ylabel('sons height (inches)');
title('Father-Son Heights');

% figure 1 subplot 2: cost function
subplot(2,2,2);
surf(theta0, theta1, J_vals)
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
plot(thetaNorm(1), thetaNorm(2), 'bx', 'MarkerSize', 20, 'LineWidth', 4)  % normal theta values
plot(thetaGrad(1), thetaGrad(2), 'gx', 'MarkerSize', 20, 'LineWidth', 4); % grad descent theta values
xlabel('\theta_0'); 
ylabel('\theta_1');
title('MSE Cost Contours');

figure 2;
% figure 2 subplot 1: cost function
subplot(2,2,1);
surf(theta0, theta1, J_vals)
xlabel('\theta_0'); 
ylabel('\theta_1');
zlabel('Cost');
title('MSE Cost');

% figure 2 subplot 2: convergence rate
subplot(2,2,2);
plot(0:1:itterations, [costGrad0 Jpast], 'b');
title('Gradient Descent Convergence');

% figure 2 subplot 3: contour plot
subplot(2,2,3);
contour(theta0, theta1, J_vals, logspace(-8, 10, 20))
hold on;
plot(thetaGrad(1), thetaGrad(2), 'gx', 'MarkerSize', 20, 'LineWidth', 4); % grad descent theta values
xlabel('\theta_0'); 
ylabel('\theta_1');
title('MSE Cost Contours');



