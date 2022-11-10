%MADALINE structure using MR2 algorithm with 2 input, 1 hidden layer, and 1
%output using majority function

close all; clear; clc;
load('Two_moons_no_overlap2.mat') %change this to desired plot

b = [rand(1) * .1, rand(1) * .1,rand(1) * .1, -.5, -1.5]; %[b1,b2,b3,b4] b4 is output bias
v1 = [.5;.5;.5]; % for y1 or function
v2 = [.5;.5;.5]; % for y2 and function
old_w = [rand(1)*.1, rand(1)*.1, rand(1)*.1; rand(1)*.1, rand(1)*.1, rand(1)*.1]; %[w11,w12,w13;w21,w22,w23] initialize weights to random small numbers
new_w = [0,0,0;0,0,0];
alpha = .000005;
delta_w = [1,1,1;1,1,1]; %[d_w11,d_w12,d_w13,d_w21,d_w22,d_w23]
y1_in = 0;
z_in= [0,0,0];
z_out = [0,0,0];
out = 0;
m = 1;
while((m < 20000)) %loop
        i = m - ceil(m/1000 -1) * 1000;
        input = X(i,:); %update input
        z_in(1) = b(1) + input(1)*old_w(1,1) + input(2)*old_w(2,1); %determines net input to hidden nodes
        z_in(2) = b(2) + input(1)*old_w(1,2) + input(2)*old_w(2,2);
        z_in(3) = b(3) + input(1)*old_w(1,3) + input(2)*old_w(2,3);
        %determine activation of each hidden node
        if(z_in(1) >= 0)
            z_out(1) = 1; 
        else
            z_out(1) = -1; 
        end

        if(z_in(2) >= 0)
            z_out(2) = 1; 
        else
            z_out(2) = -1; 
        end
        
        if(z_in(3) >= 0)
            z_out(3) = 1; 
        else
            z_out(3) = -1; 
        end

        y1_in = b(4) + v1' * z_out';%calculate y_in and y_out using activation function
        if (y1_in >= 0)
            y1_out = 1;
        else
            y1_out = -1;
        end

        y2_in = b(5) + v2' * z_out';%calculate y_in and y_out using activation function
        if (y2_in >= 0)
            y2_out = 1;
        else
            y2_out = -1;
        end

        y_out = y1_out * y2_out; % arbitrary function, if y1 and y2 are the same, then psoitive output, if different then negative

        if(y_out ~= Y(i)) % if there is error in output
            error = (Y(i) - z_in); %t-zin is the error
            %update delta weights
            delta_w = alpha * input' * error; %2 x 3 matrix of delta w in same order as other weight matrices
            if(abs(min(z_in)) < .1) % if smallest z in is within range, do algorithm
                temp = z_in; % temporary vector to manipulate
                loop = 1;
                while(loop <= 3) % loop through all values
                    if(abs(min(temp)) < .1) % if smallest z in is within range, again, redundant 
                        index = find(z_in == min(temp)); % get index of min value and store it
                        temp_Z_out = z_out; % have temp z out vector to test change
                        temp_Z_out(index) = temp_Z_out(index) * -1; % flip value of given z_out associated with min z_in
                        temp_y1_in = b(4) + v1' * temp_Z_out'; % calculate new temporary y_in
                        if (temp_y1_in > 0) % calculate temporary y_out
                            temp_y1_out = 1;
                        else
                            temp_y1_out = -1;
                        end
                        temp_y2_in = b(5) + v2' * temp_Z_out'; % calculate new temporary y_in
                        if (temp_y2_in > 0) % calculate temporary y_out
                            temp_y2_out = 1;
                        else
                            temp_y2_out = -1;
                        end
                        temp_y_out = temp_y1_out * temp_y2_out;
                        
                        if(abs(Y(i) - y_out) > abs(Y(i) - temp_y_out)) % if original y_out has more error than new y_out, do delta rule on node (index)
                            new_w(1,index) = old_w(1,index) + alpha * (temp_Z_out(index) - z_in(index)) * input(1);
                            new_w(2,index) = old_w(2,index) + alpha * (temp_Z_out(index) - z_in(index)) * input(2);
                            b(1,index) = b(1,index) + alpha * (temp_Z_out(index) - z_in(index));
                            old_w = new_w; %set new calculated weights to current weights
                        end
                        temp(index) = 1; %change the value of the min value so we dont iterate it again.
                        %recalculate y_out and if it matches dont loop
                        z_in(1) = b(1) + input(1)*old_w(1,1) + input(2)*old_w(2,1); %determines net input to hidden nodes
                        z_in(2) = b(2) + input(1)*old_w(1,2) + input(2)*old_w(2,2);
                        z_in(3) = b(3) + input(1)*old_w(1,3) + input(2)*old_w(2,3);
                        %determine activation of each hidden node
                        if(z_in(1) >= 0)
                            z_out(1) = 1; 
                        else
                            z_out(1) = -1; 
                        end
                        if(z_in(2) >= 0)
                            z_out(2) = 1; 
                        else
                            z_out(2) = -1; 
                        end
                        if(z_in(3) >= 0)
                            z_out(3) = 1; 
                        else
                            z_out(3) = -1; 
                        end
                        y1_in = b(4) + v1' * z_out';%calculate y_in and y_out using activation function
                        if (y1_in >= 0)
                            y1_out = 1;
                        else
                            y1_out = -1;
                        end
                        y2_in = b(5) + v2' * z_out';%calculate y_in and y_out using activation function
                        if (y2_in >= 0)
                            y2_out = 1;
                        else
                            y2_out = -1;
                        end
                        y_out = y1_out * y2_out;
                        if(y_out == Y(i))
                            loop = 4; % ends while statement
                        end
                    end
                    loop = loop + 1;%otherwise do next smallest value
                end
            else %else do delta rule
                for k = 1:3 %rows
                    for j = 1:2%cols
                        new_w(j,k) = old_w(j,k) + alpha * error(1,k) * input(j);
                    end
                    b(1,k) = b(1,k) + alpha * error(1,k);
                end
            end
        else
            new_w = old_w; % new weights are the same as old weight
        end
        disp(Y(i) - y_out)
        %disp(m)
        m = m+1;
        old_w = new_w;
end

%boundary equations
x = -20:29; % Good region to show boundary line
y1 = (-new_w(1,1) * x - b(1)) / new_w(2,1);
y2 = (-new_w(1,2) * x - b(2)) / new_w(2,2);
y3 = (-new_w(1,3) * x - b(3)) / new_w(2,3);



figure(1)
scatter(X(:,1),X(:,2))
hold on
plot(x,y1)
plot(x,y2)
plot(x,y3)
