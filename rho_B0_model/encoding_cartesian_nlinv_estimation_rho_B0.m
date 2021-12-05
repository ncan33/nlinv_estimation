function out = encoding_cartesian_nlinv_estimation_rho_B0(in, x, s, w, TEs, alpha_n, transpose_indicator)
% Written by Nam Gyun Lee
% Email: nmgyunl@usc.edu, ggang56@gmail.com (preferred)
% Started: 05/12/2021, Last modified: 05/15/2021

%% Declare a persistent variable
persistent cg_iter;
if isempty(cg_iter)
    cg_iter = 0;
end

%% Determine the operator type
if strcmp(transpose_indicator, 'transp')
    operator_type = 'adjoint';
elseif strcmp(transpose_indicator, 'notransp')
    operator_type = 'forward';
    cg_iter = cg_iter + 1;
end

%% Define dimensions
N1 = size(x,1);
N2 = size(x,2);

%% Calculate the forward or adjoint operator
%--------------------------------------------------------------------------
% (DG^H(xhat_n) * DG(xhat_n) + alpha_n * I) * dxhat = ...
% DG^H(xhat_n) * (y - G(xhat_n)) + alpha_n * (xhat_0 - xhat_n)
% (W^H * DF^H(x_n) * DF(x_n) * W + alpha_n * I) * dxhat = ...
% W^H * DF^H(x_n) * (y - F(x_n)) + alpha_n * (xhat_0 - xhat_n)
% (W^H * DF^H(x_n) * DF(x_n) * W + alpha_n * I) * dxhat = b <=> A * x = b
%--------------------------------------------------------------------------
tic; fprintf('(CG=%2d): Calculating the %s operator... ', cg_iter, operator_type);
dxhat = reshape(in, [N1 N2 2]);

%--------------------------------------------------------------------------
% Calculate dx = W * dxhat (N1 x N2 x 2)
%--------------------------------------------------------------------------
dx = W(dxhat, s, w);

%sqrt(reshape(sum(sum(abs(dx).^2,1),2),[2 1]))

%--------------------------------------------------------------------------
% Calculate DF(x_n) * dx (N1 x N2 x Ne)
%--------------------------------------------------------------------------
dy = DF(x, dx, TEs);

%--------------------------------------------------------------------------
% Calculate DF^H(x_n) * (DF(x_n) * dx) (N1 x N2 x 2)
%--------------------------------------------------------------------------
out = DF_adjoint(x, dy, TEs);

%--------------------------------------------------------------------------
% Calculate W^H * (DF^H(x_n) * DF(x_n) * dx) (N1 x N2 x 2)
%--------------------------------------------------------------------------
out = W_adjoint(out, s, w);

%--------------------------------------------------------------------------
% Add alpha_n * dxhat (N1 x N2 x 2)
%--------------------------------------------------------------------------
out = out + alpha_n * dxhat;

%--------------------------------------------------------------------------
% Vectorize the output
%--------------------------------------------------------------------------
out = out(:);
fprintf('done! (%6.4f sec)\n', toc);

end