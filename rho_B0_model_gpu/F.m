function y = F(x, TEs)
% x    : N1 x N2 x 2
% TEs  : Ne x 1         [sec]
% y    : N1 x N2 x Ne
%
% Written by Nam Gyun Lee
% Email: nmgyunl@usc.edu, ggang56@gmail.com (preferred)
% Started: 05/12/2021, Last modified: 05/15/2021

%--------------------------------------------------------------------------
% Calculate F(x)
% F_{j,m}(x) = rho * exp(1j * 2 * pi * f_B0 * TE_m)
% with x = [rho; f_B0]
%--------------------------------------------------------------------------
N1 = size(x,1);
N2 = size(x,2);
Ne = length(TEs);

y = complex(zeros(N1, N2, Ne, 'double', 'gpuArray'));
for m = 1:Ne
    %----------------------------------------------------------------------
    % Calculate rho * exp(1j * 2 * pi * f_B0 * TE_m)
    %----------------------------------------------------------------------
    y(:,:,m) = x(:,:,1) .* exp(1j * 2 * pi * x(:,:,2) * TEs(m));
end

end