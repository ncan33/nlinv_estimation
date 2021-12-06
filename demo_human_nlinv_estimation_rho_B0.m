% demo_human_nlinv_estimation_rho_B0.m
% Written by Nam Gyun Lee
% Email: namgyunl@usc.edu, ggang56@gmail.com (preferred)
% Started: 05/13/2021, Last modified: 05/17/2021

%% Clean slate
close all; clear all; clc;

%% Set directory names
%--------------------------------------------------------------------------
% Package directory
%--------------------------------------------------------------------------
computer_type = computer;
if strcmp(computer_type, 'PCWIN64')
    src_directory = 'E:\nlinv_estimation\rho_B0_model';
elseif strcmp(computer_type, 'GLNXA64')
    src_directory = '/server/home/nlee/nlinv_estimation/rho_B0_model';
end

%% Add paths
addpath(genpath(src_directory));

%% Load image and sensitivity maps to create multi-channel images
image_ori = 'axial';
multiecho_filename = sprintf('human_%s_data', image_ori); 
load(multiecho_filename);
[N1,N2,N3,Ne] = size(im_echo);

%% Define a function handle
if strcmp(image_ori, 'sagittal') % sagittal plane
    reorient = @(x) x;
elseif strcmp(image_ori, 'coronal') % coronal plane
    reorient = @(x) x;
elseif strcmp(image_ori, 'axial') % transverse plane
    reorient = @(x) flip(rot90(x, -1), 2);
end

%% Calculate the regularization term for the B0 field inhomogeneity
w_fB0 = 32;  % 32 in BART?
h_fB0 = 16;  % Sobolev index for B0 field inhomogeneity

weights = zeros(N1, N2, 'double');
for idx2 = 1:N2
    for idx1 = 1:N1
        %------------------------------------------------------------------
        % Calculate the k-space weight for B0 field inhomogeneity
        %------------------------------------------------------------------
        kx = (-floor(N1/2) + idx1 - 1) / N1;
        ky = (-floor(N2/2) + idx2 - 1) / N2;
        weights(idx1,idx2) = 1 / (1 + w_fB0 * (kx^2 + ky^2))^h_fB0;
    end
end

%% Calculate the scaling matrix
scaling = ones(2, 1, 'double');

for slice_nr = 1:N3
    %% Normalize the data vector
    y = reshape(im_echo(:,:,slice_nr,:), [N1 N2 Ne]);
    scale_factor = 100 / norm(y(:));
    y_scaled = y * scale_factor;

    %% Set an initial guess xhat0 (N1 x N2 x 2)
    xhat0 = complex(zeros(N1, N2, 2, 'double'));
    xhat0(:,:,1) = 1; % rhohat
    xhat = xhat0;

    %% Calculate x0 = W * xhat0 (N1 x N2 x 2)
    x0 = W(xhat0, scaling, weights);
    x = x0;

    %% Set initial parameters
    alpha0 = 1;
    q = 2 / 3;
    alpha_min = 1e-6;

    %% Perform the IRGNM algorithm
    irgnm_iterations = 60;  % maximum number of Gauss-Newton iterations
    cg_iterations = 250;    % maximum number of CG iterations
    limit = 1e-10;          % tolerance of LSQR

    rho_update    = complex(zeros(N1, N2, irgnm_iterations, 'double'));
    fB0_update    = complex(zeros(N1, N2, irgnm_iterations, 'double'));
    alpha_update  = zeros(irgnm_iterations, 1, 'double');
    rho_norm      = zeros(irgnm_iterations, 1, 'double');
    fB0_norm      = zeros(irgnm_iterations, 1, 'double');
    residual_norm = zeros(irgnm_iterations, 1, 'double');

    start_time = tic;
    for n = 0:irgnm_iterations-1
        clear encoding_cartesian_nlinv_estimation_rho_B0;

        fprintf('(n=%d/%d): Performing NLINV rho/B0 model estimation...\n', n, irgnm_iterations-1);
        %------------------------------------------------------------------
        % Set alpha_n
        % A minimum value of alpha is introduced to control the noise in
        % higher Gauss-Newton steps
        %------------------------------------------------------------------
        alpha = max(alpha_min, alpha0 * q^n);

        %------------------------------------------------------------------
        % Calculate the right side of the normal equation
        % (DG^H(xhat_n) * DG(xhat_n) + alpha_n * I) * dxhat = ...
        % DG^H(xhat_n) * (y - G(xhat_n)) + alpha_n * (xhat_0 - xhat_n)
        % (W^H * DF^H(x_n) * DF(x_n) * W + alpha_n * I) * dxhat = ...
        % W^H * DF^H(x_n) * (y - F(x_n)) + alpha_n * (xhat_0 - xhat_n)
        %------------------------------------------------------------------
        residual = y_scaled - F(x, TEs);
        b = DF_adjoint(x, residual, TEs); % N1 x N2 x 2
        b = W_adjoint(b, scaling, weights); % N1 x N2 x 2
        b = b + alpha * (xhat0 - xhat);
        b = b(:);

        %------------------------------------------------------------------
        % Calculate an approximate solution to the linearized problem using
        % the conjugate gradient algorithm
        %------------------------------------------------------------------
        tstart = tic;
        E = @(in,tr) encoding_cartesian_nlinv_estimation_rho_B0(in, x, scaling, weights, TEs, alpha, tr);
        [dxhat, flag, relres, iter, resvec, lsvec] = lsqr(E, b, limit, cg_iterations, [], [], []);
        dxhat = reshape(dxhat, [N1 N2 2]);
        telapsed = toc(tstart);

        %------------------------------------------------------------------
        % Update the solution: xhat_(n+1) = xhat_n + dxhat
        %------------------------------------------------------------------
        xhat = xhat + dxhat;
        x = W(xhat, scaling, weights);
        fprintf('done! (%6.4f/%6.4f sec)\n', telapsed, toc(start_time));

        %------------------------------------------------------------------
        % Save intermediate results
        %------------------------------------------------------------------
        rho_update(:,:,n+1) = x(:,:,1);
        fB0_update(:,:,n+1) = x(:,:,2);
        alpha_update(n+1)   = alpha;

        rho_norm(n+1)      = norm(vec(x(:,:,1)));
        fB0_norm(n+1)      = norm(vec(x(:,:,2)));
        residual_norm(n+1) = norm(vec(residual));

        %------------------------------------------------------------------
        % Display results
        %------------------------------------------------------------------
        figure('Color', 'w', 'Position', [-5 388 560 420]);
        imagesc(abs(reorient(rho_update(:,:,n+1)))); axis image;
        title(sprintf('Iteration n = %d, alpha = %e', n, alpha));
        colormap(gray(256));
        colorbar;

        figure('Color', 'w', 'Position', [1047 395 560 420]);
        imagesc(reorient(fB0_update(:,:,n+1))); axis image;
        title(sprintf('Iteration n = %d, alpha = %e', n, alpha));
        colormap(jet(256));
        colorbar;
        drawnow;
    end

    %% Postprocessing
    rho_final = x(:,:,1) / scale_factor;
    fB0_final = x(:,:,2);

    %% Save results
    save(sprintf('human_%s_slice%d_min%3.1e.mat', image_ori, slice_nr, alpha_min), ...
        'rho_update', 'fB0_update', 'alpha_update', ...
        'rho_norm', 'fB0_norm', 'residual_norm', ...
        'scale_factor', 'w_fB0', 'alpha_min', 'irgnm_iterations');

end