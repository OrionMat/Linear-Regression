function [theta, Jpast] = gradDescent (X, y, theta, alpha, m, itterations)
  
  for i = 1:itterations
    theta = theta - (alpha/m) * X'*(X*theta-y);
    Jpast(i) = costMSE(X, theta, y, m);
  end 
  
endfunction
