% demo_human_nlinv_estimation_rho_B0.m
% Written by Nam Gyun Lee
% Email: namgyunl@usc.edu, ggang56@gmail.com (preferred)
% Started: 05/13/2021, Last modified: 05/17/2021

%% Clean slate
close all; clear all; clc;

%% Set source directories
computer_type = computer;
if strcmp(computer_type, 'PCWIN64')
    src_directory = 'E:\nlinv_estimation';
elseif strcmp(computer_type, 'GLNXA64')
    src_directory = '/server/home/nlee/nlinv_estimation';
end

%% Add source directories to search path
addpath(genpath(fullfile(src_directory, 'rho_B0_model')));

%% Load multi-echo images
image_ori = 'sagittal';
osf = 2;
multiecho_fullpath = fullfile(src_directory, sprintf('human_%s_data_osf%d_gpu', image_ori, osf)); 
load(multiecho_fullpath);
[N1,N2,Ns,Ne] = size(im_echo);

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

%% Set initial parameters
alpha0 = 1;
q = 2 / 3;
alpha_min = 1e-6;

% data without oversampling
irgnm_iterations = 60;  % maximum number of Gauss-Newton iterations
cg_iterations = 250;    % maximum number of CG iterations
tol = 1e-10;            % tolerance of LSQR

% data with oversampling
irgnm_iterations = 150; % maximum number of Gauss-Newton iterations
cg_iterations = 250;    % maximum number of CG iterations

%% Make output directory
output_directory = fullfile(src_directory, sprintf('nlinv_estimation_rho_B0_model_human_%s_osf%d_min%3.1e', image_ori, osf, alpha_min));
mkdir(output_directory);

%% Perform NLINV rho/B0 estimation per slice
for slice_nr = 1:Ns
    %% Normalize the data vector
    y = reshape(im_echo(:,:,slice_nr,:), [N1 N2 Ne]);
    scale_factor = 100 / norm(y(:)); % 100 * 1000 / (2*pi)
    y_scaled = y * scale_factor;

    %% Set an initial guess xhat0 (N1 x N2 x 2)
    xhat0 = complex(zeros(N1, N2, 2, 'double'));
    xhat0(:,:,1) = 1; % rhohat
    xhat = xhat0;

    %% Calculate x0 = W * xhat0 (N1 x N2 x 2)
    x0 = W(xhat0, scaling, weights);
    x = x0;

    %% Perform the IRGNM algorithm
    rho_update    = complex(zeros(N1, N2, irgnm_iterations, 'double'));
    fB0_update    = complex(zeros(N1, N2, irgnm_iterations, 'double'));
    alpha_update  = zeros(irgnm_iterations, 1, 'double');
    rho_norm      = zeros(irgnm_iterations, 1, 'double');
    fB0_norm      = zeros(irgnm_iterations, 1, 'double');
    residual_norm = zeros(irgnm_iterations, 1, 'double');

    start_time = tic;
    for n = 0:irgnm_iterations-1
        clear encoding_cartesian_nlinv_estimation_rho_B0;

        fprintf('(%d/%d),(n=%d/%d): Performing NLINV rho/B0 estimation... ', slice_nr, Ns, n, irgnm_iterations-1);
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
        [dxhat, flag, relres, iter, resvec, lsvec] = lsqr(E, b, tol, cg_iterations, [], [], []);
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
        % Stopping criterion
        %------------------------------------------------------------------
        rel_norm = abs(rho_norm(n+1) - rho_norm(n)) / rho_norm(n);
        if rel_norm < 1e-6
            break;
        end

        %------------------------------------------------------------------
        % Display results
        %------------------------------------------------------------------
        if 0
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
    end
    computation_time = toc(start_time);

    %% Postprocessing
    rho_final = rho_update(:,:,irgnm_iterations) / scale_factor;
    fB0_final = fB0_update(:,:,irgnm_iterations);

    %% Save results
    output_fullpath = fullfile(output_directory, sprintf('nlinv_estimation_rho_B0_model_human_%s_osf%d_slice%d_min%3.1e.mat', image_ori, osf, slice_nr, alpha_min));
    tstart = tic; fprintf('Saving results: %s... ', output_fullpath);
    save(output_fullpath, ...
        'rho_update', 'fB0_update', 'alpha_update', ...
        'rho_norm', 'fB0_norm', 'residual_norm', ...
        'rho_final', 'fB0_final', 'scale_factor', ...
        'w_fB0', 'h_fB0', 'alpha_min', 'irgnm_iterations', 'slice_nr', 'computation_time', '-v7.3');
    fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
end