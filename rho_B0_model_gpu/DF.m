function dy = DF(x, dx, TEs)
% x    : N1 x N2 x 2
% dx   : N1 x N2 x 2
% TEs  : Ne x 1
% dy   : N1 x N2 x Ne
%
% Written by Nam Gyun Lee
% Email: nmgyunl@usc.edu, ggang56@gmail.com (preferred)
% Started: 05/12/2021, Last modified: 05/15/2021

%--------------------------------------------------------------------------
% Calculate dy = DF(x) * dx (N1 x N2 x Ne)
%--------------------------------------------------------------------------
N1 = size(x,1);
N2 = size(x,2);
Ne = length(TEs);

dy = complex(zeros(N1, N2, Ne, 'double', 'gpuArray'));
for m = 1:Ne
    %----------------------------------------------------------------------
    % Calculate psi_m (N1 x N2)
    %----------------------------------------------------------------------
    psi_m = exp(1j * 2 * pi * x(:,:,2) * TEs(m));

    %----------------------------------------------------------------------
    % Calculate diag(psi_m) * drho + (1j * 2 * pi * TE_m) * diag(rho) * diag(psi_m) * df_B0
    %----------------------------------------------------------------------
    dy(:,:,m) = psi_m .* dx(:,:,1) + (1j * 2 * pi * TEs(m)) * (x(:,:,1) .* psi_m .* dx(:,:,2)); 
end

end