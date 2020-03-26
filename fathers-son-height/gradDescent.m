function [theta, J_hist] = gradDescent (X, theta, y, alpha, m, itterations)
  for i = 1:itterations
    theta = theta - (alpha/m) * X'*(X*theta-y);
    J_hist(i) = costMSE(X, theta, y, m);
  end
endfunction
