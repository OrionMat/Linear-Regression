function J = costMSE (X, theta, y, m)
  avg = 1/(2*m);
  errors = X*theta - y;
  J = avg * errors' * errors;
endfunction
