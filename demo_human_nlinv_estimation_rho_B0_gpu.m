% demo_human_nlinv_estimation_rho_B0_gpu.m
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
addpath(genpath(fullfile(src_directory, 'rho_B0_model_gpu')));

%% Load multi-echo images
image_ori = 'sagittal';
remove_oversampling = 1;
multiecho_fullpath = fullfile(src_directory, sprintf('human_%s_data_ro%d', image_ori, remove_oversampling)); 
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

%% Set initial parameters
alpha0 = 1;
q = 2 / 3;

%% Set reconstruction parameters
alpha_min = 1e-6;        % minimum regularization parameter
irgnm_iterations = 150;  % maximum number of Gauss-Newton iterations
cg_iterations = 250;     % maximum number of CG iterations
tol = 1e-10;             % tolerance of LSQR

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

%% Copy data only once on workers in a parallel pool
%--------------------------------------------------------------------------
% Determine the number of GPU devices
%--------------------------------------------------------------------------
nr_GPUs = gpuDeviceCount("available");

%--------------------------------------------------------------------------
% Create a parallel pool of workders
%--------------------------------------------------------------------------
delete(gcp);
parpool('local', nr_GPUs);

%--------------------------------------------------------------------------
% Copy data only once on workers in a parallel pool
%--------------------------------------------------------------------------
tstart = tic; fprintf('Copying data only once on workers in parallel pool...\n');
im_echo_device = parallel.pool.Constant(@() gpuArray(im_echo));
TEs_device     = parallel.pool.Constant(@() gpuArray(TEs));
weights_device = parallel.pool.Constant(@() gpuArray(weights));
scaling_device = parallel.pool.Constant(@() gpuArray(scaling));
fprintf('done! (%6.4f sec)\n', toc(tstart));

%% Make output directory
output_filename = sprintf('nlinv_estimation_rho_B0_model_human_%s_osf%d_min%3.1e_tol%3.1e_w%d_iter%d_gpu', image_ori, remove_oversampling, alpha_min, tol, w_fB0, irgnm_iterations);
output_directory = fullfile(src_directory, output_filename);
mkdir(output_directory);

%% Perform NLINV rho/B0 estimation per slice
parfor slice_nr = 1:Ns
    %% Normalize the data vector
    y = reshape(im_echo_device.Value(:,:,slice_nr,:), [N1 N2 Ne]);
    scale_factor_device = 100 / norm(y(:));
    y_scaled = y * scale_factor_device;

    %% Set an initial guess xhat0 (N1 x N2 x 2)
    xhat0 = complex(zeros(N1, N2, 2, 'double', 'gpuArray'));
    xhat0(:,:,1) = 1; % rhohat
    xhat = xhat0;

    %% Calculate x0 = W * xhat0 (N1 x N2 x 2)
    x0 = W(xhat0, scaling_device.Value, weights_device.Value);
    x = x0;

    %% Perform the IRGNM algorithm
    rho_update_device    = complex(zeros(N1, N2, irgnm_iterations, 'double', 'gpuArray'));
    fB0_update_device    = complex(zeros(N1, N2, irgnm_iterations, 'double', 'gpuArray'));
    alpha_update_device  = zeros(irgnm_iterations, 1, 'double', 'gpuArray');
    rho_norm_device      = zeros(irgnm_iterations, 1, 'double', 'gpuArray');
    fB0_norm_device      = zeros(irgnm_iterations, 1, 'double', 'gpuArray');
    residual_norm_device = zeros(irgnm_iterations, 1, 'double', 'gpuArray');

    start_time = tic;
    for n = 0:irgnm_iterations-1
        %clear encoding_cartesian_nlinv_estimation_rho_B0;

        fprintf('(%2d/%2d),(n=%3d/%3d): IRGNM... ', slice_nr, Ns, n, irgnm_iterations-1);
        %------------------------------------------------------------------
        % Set alpha_n
        % A minimum value of alpha is introduced to control the noise in
        % higher Gauss-Newton steps
        %------------------------------------------------------------------
        alpha = gpuArray(max(alpha_min, alpha0 * q^n));

        %------------------------------------------------------------------
        % Calculate the right side of the normal equation
        % (DG^H(xhat_n) * DG(xhat_n) + alpha_n * I) * dxhat = ...
        % DG^H(xhat_n) * (y - G(xhat_n)) + alpha_n * (xhat_0 - xhat_n)
        % (W^H * DF^H(x_n) * DF(x_n) * W + alpha_n * I) * dxhat = ...
        % W^H * DF^H(x_n) * (y - F(x_n)) + alpha_n * (xhat_0 - xhat_n)
        %------------------------------------------------------------------
        residual = y_scaled - F(x, TEs_device.Value);
        b = DF_adjoint(x, residual, TEs_device.Value); % N1 x N2 x 2
        b = W_adjoint(b, scaling_device.Value, weights_device.Value); % N1 x N2 x 2
        b = b + alpha * (xhat0 - xhat);
        b = b(:);

        %------------------------------------------------------------------
        % Calculate an approximate solution to the linearized problem using
        % the conjugate gradient algorithm
        %------------------------------------------------------------------
        tstart = tic;
        E = @(in,tr) encoding_cartesian_nlinv_estimation_rho_B0(in, x, scaling_device.Value, weights_device.Value, TEs_device.Value, alpha, tr);
        [dxhat, flag, relres, iter, resvec] = lsqr(E, b, tol, cg_iterations, [], [], []);
        dxhat = reshape(dxhat, [N1 N2 2]);
        telapsed = toc(tstart);

        %------------------------------------------------------------------
        % Update the solution: xhat_(n+1) = xhat_n + dxhat
        %------------------------------------------------------------------
        xhat = xhat + dxhat;
        x = W(xhat, scaling_device.Value, weights_device.Value);
        fprintf('res=%e,alpha=%e,(flag=%d,CG=%3d) done! (%6.4f/%6.4f sec)\n', norm(vec(residual)), alpha, flag, iter, telapsed, toc(start_time));

        %------------------------------------------------------------------
        % Save intermediate results
        %------------------------------------------------------------------
        rho_update_device(:,:,n+1) = x(:,:,1);
        fB0_update_device(:,:,n+1) = x(:,:,2);
        alpha_update_device(n+1)   = alpha;

        rho_norm_device(n+1)      = norm(vec(x(:,:,1)));
        fB0_norm_device(n+1)      = norm(vec(x(:,:,2)));
        residual_norm_device(n+1) = norm(vec(residual));
    end
    computation_time = toc(start_time);

    %% Transfer arrays from the GPU to the CPU
    rho_update    = gather(rho_update_device);
    fB0_update    = gather(fB0_update_device);
    alpha_update  = gather(alpha_update_device);
    rho_norm      = gather(rho_norm_device);
    fB0_norm      = gather(fB0_norm_device);
    residual_norm = gather(residual_norm_device);
    scale_factor  = gather(scale_factor_device);

    %% Postprocessing
    rho_final = rho_update(:,:,irgnm_iterations) / scale_factor;
    fB0_final = fB0_update(:,:,irgnm_iterations);

    %% Save results    
    output_fullpath = fullfile(output_directory, sprintf('nlinv_estimation_rho_B0_model_human_%s_ro%d_slice%d_min%3.1e_tol%3.1e_w%d_iter%d_gpu.mat', image_ori, remove_oversampling, slice_nr, alpha_min, tol, w_fB0, irgnm_iterations));
    tstart = tic; fprintf('Saving results: %s... ', output_fullpath);
    parsave_rho_B0(output_fullpath, alpha_update, rho_norm, fB0_norm, residual_norm, rho_final, fB0_final, scale_factor, w_fB0, h_fB0, alpha_min, irgnm_iterations, slice_nr, computation_time);
    fprintf('done! (%6.4f/%6.4f sec)\n', toc(tstart), toc(start_time));
end