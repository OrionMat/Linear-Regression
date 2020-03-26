function J_vals = costValues(X, y, theta0, theta1, m)

  J_vals = zeros(length(theta0), length(theta1));
  for i = 1:length(theta0)
      for j = 1:length(theta1)
        t = [theta0(i); theta1(j)];
        J_vals(i,j) = costMSE(X, t, y, m);
      end
  end

  J_vals = J_vals';

endfunction
