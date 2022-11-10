function [delta_w, delta_bias] = backPropagationHidden(output_error,w_out,input,z_in,eta)
    delta_w = zeros(20,2); % initialize all variables
    delta_bias = zeros(20,1);
    error_in = zeros(20,1);
    error = zeros(20,1);
    for i = 1:size(w_out,1) % loop through all hidden nodes
        error_in(i) = output_error * w_out(i); %error at node is sum of weighted errors in
        error(i) = error_in(i) * sech(z_in(i)); % error value is error_in * f'(y_in)
        delta_w(i,1) = eta * error(i) * input(1,1); % delta for first first input and hidden node i
        delta_w(i,2) = eta * error(i) * input(1,2); % delta for second first input and hidden node i
        delta_bias(i) = eta * error(i); % delta for bias of hidden node i
    end
end