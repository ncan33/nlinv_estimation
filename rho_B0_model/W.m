function x = W(xhat, s, w)
% xhat: N1 x N2 x 2
% s   : 2 x 1
% w   : N1 x N2
% x   : N1 x N2 x 2
%
% Written by Nam Gyun Lee
% Email: nmgyunl@usc.edu, ggang56@gmail.com (preferred)
% Started: 05/12/2021, Last modified: 05/15/2021

%--------------------------------------------------------------------------
% Calculate x = T * S * xhat, where w = 1 / (1 + w_. * abs(k))^h_.
%  
% S = [s(1)     ]
%     [     s(2)]
%
% T = [I                    ]
%     [    F^-1 * diag(w_B0)]
%--------------------------------------------------------------------------
N1 = size(xhat,1);
N2 = size(xhat,2);
x = complex(zeros(N1, N2, 2, 'double'));

x(:,:,1) = s(1) * xhat(:,:,1); % rho
x(:,:,2) = ifft2c((s(2) * xhat(:,:,2)) .* w(:,:,1)); % f_B0

end