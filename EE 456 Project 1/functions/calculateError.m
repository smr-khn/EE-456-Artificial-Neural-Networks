function error = calculateError(output, target)
    instant_error = target - output; % error is target - output
    error = .5 * instant_error ^2; % instantaneous error energy
end