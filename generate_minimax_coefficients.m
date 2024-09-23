% ------------
% Generate coefficients for a MiniMax optimized fractional delay filter
%
% Implementation based on:
% W. Putnam and J. Smith, “Design of fractional delay filters using convex
% optimization," in Proceedings of 1997 Workshop on Applications of
% Signal Processing to Audio and Acoustics, New Paltz, NY, USA, 1997
% https://ccrma.stanford.edu/~jos/resample/optfir.pdf
%
% A. Carson 2024
% -------------


clc
clear all
close all

delay = 44.1/48 - 1;                        % delay [samples]
orders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];   % interp orders
M = 1024;                                   % num freq points
bw = 0.5;                                   % optimisation bandwidth (Nyquist = 1.0)
coeffs = zeros(length(orders), length(orders)+1);


for oo = 1:length(orders)
    order = orders(oo);
    kernel = frac_delay_soco(delay, order+1, M, bw, true);
    kernel = kernel / sum(kernel);    % scale to ensure unity sum           
    coeffs(oo, 1:length(kernel)) = kernel;
end

savename = sprintf('lookup_tables/L_inf_delta=%.3f_bw=%.1f.csv', delay, bw)
writematrix(coeffs, savename)


function x = frac_delay_soco(delay, N, M, max_freq, unity_sum)
    %
    % Get fir coefficients for minimax fractional delay filter
    % 
    % N: filter order
    % M: num frequency samples
    % max_freq: relative to Nyquist
    % unity_sum: add constraint to ensure coefficients sum to 1
    
    % freq ax
    n = (0:N-1);                            % filter coefficient idx
    omega = pi * max_freq * (0:M-1)'/M;     % discrete frequencies 
    f = [zeros(N, 1); 1];                   
    gamma = 0;      
    H_d = exp(-1j * delay * omega);         % ideal response
    
    % dummy init
    socs = [secondordercone(0, 0, 0, 0)];
    
    % construct second order cone optimization 
    for i = 1:M
        a_T = exp(-1j * n * omega(i));
        A = [real(a_T), 0; imag(a_T), 0];
        
        b = [real(H_d(i)); imag(H_d(i))];
        socs(i) = secondordercone(A,b,f,gamma);
    end
    
    if unity_sum
        Aeq = [ones(1, N), 0];
        beq = 1;    
        x = coneprog(f, socs, Aeq, beq, Aeq, beq); % additional constraint to ensure coeffs sum to 1
    else
        x = coneprog(f, socs);
    end
    
    x = x(1:end-1)';
end
