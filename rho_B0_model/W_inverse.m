function xhat = W_inverse(x, s, w)
% x   : N1 x N2 x 2
% s   : 2 x 1
% w   : N1 x N2
% xhat: N1 x N2 x 2

%--------------------------------------------------------------------------
% Calculate xhat = S^-1 * T^-1 * x, where w = 1 / (1 + w_. * abs(k))^h_.
%
% S^-1 = [1/s(1)       ]
%        [       1/s(2)]
%
% T^-1 = [I                 ]
%        [  diag(1/w_B0) * F]
%--------------------------------------------------------------------------
N1 = size(x,1);
N2 = size(x,2);
xhat = complex(zeros(N1, N2, 2, 'double'));

xhat(:,:,1) = x(:,:,1) / s(1); % rhohat_W
xhat(:,:,2) = fft2c(x(:,:,2)) ./ w(:,:,1) / s(2); % fhat_B0

end