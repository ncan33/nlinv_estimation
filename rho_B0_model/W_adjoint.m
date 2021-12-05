function xhat = W_adjoint(x, s, w)
% x   : N1 x N2 x 2
% s   : 2 x 1
% w   : N1 x N2
% xhat: N1 x N2 x 2

%--------------------------------------------------------------------------
% Calculate xhat = S^H * T^H * x, where w = 1 / (1 + w_. * abs(k))^h_.
%
% S^H = [s(1)     ]
%       [     s(2)]
%
% T^H = [I               ]
%       [  diag(w_B0) * F]
%--------------------------------------------------------------------------
N1 = size(x,1);
N2 = size(x,2);
xhat = complex(zeros(N1, N2, 2, 'double'));

xhat(:,:,1) = s(1) * x(:,:,1); % rhohat_W
xhat(:,:,2) = s(2) * (fft2c(x(:,:,2)) .* w(:,:,1)); % fhat_B0

end