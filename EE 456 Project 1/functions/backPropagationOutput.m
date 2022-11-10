function [error,delta_w,delta_bias] = backPropagationOutput(target,y,y_in,z,eta)
    error = (target - y) * sech(y_in)^2; % solve for gradient error (d-y) * f'(y_in)
    delta_w = zeros(20,1);
    for i = 1:size(z,1) % loop through all hidden nodes
        delta_w(i) = eta * error * z(i); % del_w = eta * gradient error * z_output
    end
    delta_bias = eta * error; % bias is same calc but z is 1
end