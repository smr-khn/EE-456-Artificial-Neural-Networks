function [y,y_in,z, z_in] = feedForward(input, w_hid, w_out, bias_hid, bias_out, nn_size)
    z_in = zeros(nn_size(2),1);
    z = zeros(nn_size(2),1);
    y_in = 0;
    for i = 1:nn_size(2) % loop through all hidden nodes
        z_in(i,1) = bias_hid(i) + input(1,1) * w_hid(i,1) + input(1,2) * w_hid(i,2); %calc z_in at every hidden node
        z(i) = tanh(z_in(i)); % use tanh activation function on z_in to get z
        y_in = y_in + z(i) * w_out(i,1); % add z to y_in
    end
    y_in = y_in + bias_out; % add bias term after all z's are summed
    y = tanh(y_in); % use tanh activation function
end