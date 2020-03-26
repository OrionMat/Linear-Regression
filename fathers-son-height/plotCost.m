function plotCost (X, y, m)
  
  theta0 = linspace(-20, 70, 100)';
  theta1 = linspace(-3, 3, 100)';

  J_vals = zeros(length(theta0), length(theta1));
  for i = 1:length(theta0)
      for j = 1:length(theta1)
        t = [theta0(i); theta1(j)];
        J_vals(i,j) = costMSE(X, t, y, m);
      end
  end

  J_vals = J_vals';
  
  % surface plot
  figure 2;
  surf(theta0, theta1, J_vals)
  xlabel('\theta_0'); 
  ylabel('\theta_1');

  % Contour plot
  figure 3;
  contour(theta0, theta1, J_vals, logspace(-3, 4, 20))
  xlabel('\theta_0'); ylabel('\theta_1');

endfunction
