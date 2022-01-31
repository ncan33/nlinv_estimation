% demo_process_rho_B0_mat_files.m
% Written by Nam Gyun Lee
% Email: namgyunl@usc.edu, ggang56@gmail.com (preferred)
% Started: 05/15/2021, Last modified: 05/15/2021

%% Clean slate
close all; clear all; clc;

%% Set source directories
thirdparty_directory = 'E:\nlinv_estimation\thirdparty';

%% Add source directories to search path
addpath(genpath(thirdparty_directory));

%% Define parameters
%image_ori = 'axial';
image_ori = 'sagittal';
irgnm_nr = 150; % usually 35 was good enough
hz_range = [-100 100];

%% Define data directory
nlinv_result_directory = 'E:\nlinv_estimation\nlinv_estimation_rho_B0_model_human_sagittal_ro1_min1.0e-06_tol1.0e-10_w32_iter150_gpu';
ro_loc = strfind(nlinv_result_directory, '_ro');
remove_oversampling = str2double(nlinv_result_directory(ro_loc+3:ro_loc+3));
min_loc = strfind(nlinv_result_directory, '_min');
alpha_min = str2double(nlinv_result_directory(min_loc+3:min_loc+9));
w_loc = strfind(nlinv_result_directory, '_w');
w_fB0 = str2double(nlinv_result_directory(w_loc+2:w_loc+3));

%% Load a .mat file containing multi-echo images
multiecho_fullpath = fullfile('E:\nlinv_estimation', sprintf('human_sagittal_data_ro%d', remove_oversampling));
load(multiecho_fullpath);
[N1,N2,Ns,Ne] = size(im_echo);

%% Set output directory
output_directory = nlinv_result_directory;
mkdir(output_directory);

%% Define a function handle
if strcmp(image_ori, 'sagittal') % sagittal plane
    reorient = @(x) x;
elseif strcmp(image_ori, 'coronal') % coronal plane
    reorient = @(x) x;
elseif strcmp(image_ori, 'axial') % transverse plane
    reorient = @(x) flip(rot90(x, -1), 2);
end

%% Process a mat file
dir_info = dir(fullfile(nlinv_result_directory, 'nlinv_estimation_*.mat'));
rho_nlinv = complex(zeros(N1, N2, Ns, 'double'));
B0map_nlinv = zeros(N1, N2, Ns, 'double');

for idx = 1:length(dir_info)
    %----------------------------------------------------------------------
    % Read a NLINV rho/B0 result file
    %----------------------------------------------------------------------
    mat_fullpath = fullfile(dir_info(idx).folder, dir_info(idx).name);
    load(mat_fullpath);
    rho_nlinv(:,:,slice_nr) = rho_final;
    B0map_nlinv(:,:,slice_nr) = real(fB0_final);

    %----------------------------------------------------------------------
    % L2 norm
    %----------------------------------------------------------------------
    figure('Color', 'w', 'Position', [4 351 1595 448]);
    subplot(1,3,1);
    plot((1:irgnm_iterations), rho_norm, 'b.-', 'MarkerSize', 12);
    grid on; grid minor;
    xlabel('Gauss-Newton Iteration Number');
    title(sprintf('L2 norm of rho, slice = %d', slice_nr));
    subplot(1,3,2);
    plot((1:irgnm_iterations), fB0_norm, 'b.-', 'MarkerSize', 12);
    %ylim([0 18000]);
    grid on; grid minor;
    title(sprintf('L2 norm of B0 field map, slice = %d', slice_nr));
    xlabel('Gauss-Newton Iteration Number');
    subplot(1,3,3);
    plot((1:irgnm_iterations), residual_norm, 'b.-', 'MarkerSize', 12);
    grid on; grid minor;
    xlabel('Gauss-Newton Iteration Number');
    title(sprintf('Residual norm, slice = %d', slice_nr));
    export_fig(fullfile(output_directory, sprintf('L2norm_slice%d_ro%d_min%3.1e_w%d_iter%d', slice_nr, remove_oversampling, alpha_min, w_fB0, irgnm_iterations)), '-m2', '-tif');
    close gcf;
end

%% Display all slices
c = floor(Ns/2) + 1;

for idx = 1:Ne
    echo_montage = complex(zeros(N1*2, N2*c, 'double'));
    echo_montage(1:N1,(1:N2*c)) = reshape(reorient(im_echo(:,:,1:c,idx)), [N1 N2*c]);
    echo_montage((1:N1)+N1,1:N2*(c-1)) = reshape(reorient(im_echo(:,:,c+1:end,idx)), [N1 N2*(Ns-c)]);

    %----------------------------------------------------------------------
    % Magnitude of echo images
    %----------------------------------------------------------------------
    figure('Color', 'k', 'Position', [1 1 1600 823]);
    imagesc(abs(echo_montage)); axis image;
    set(gca, 'XColor', 'w', 'YColor', 'w', 'XTick', [], 'YTick', []);
    colormap(gray(256));
    hc = colorbar;
    set(hc, 'Color', 'w', 'FontSize', 14);
    text(N2 * c / 2, 0, {sprintf('Magnitude of echo images (echo = %d)', idx)}, 'Color', 'w', 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    export_fig(fullfile(output_directory, sprintf('echo%d_mag', idx)), '-r300', '-tif'); % '-c[100,680,210,870]' [top,right,bottom,left]
    close gcf;

    %----------------------------------------------------------------------
    % Phase of echo images
    %----------------------------------------------------------------------
    figure('Color', 'k', 'Position', [1 1 1600 823]);
    imagesc(angle(echo_montage)*180/pi); axis image;
    set(gca, 'XColor', 'w', 'YColor', 'w', 'XTick', [], 'YTick', []);
    colormap(hsv(256));
    hc = colorbar;
    set(hc, 'Color', 'w', 'FontSize', 14);
    text(N2 * c / 2, 0, {sprintf('Phase of echo images (echo = %d)', idx)}, 'Color', 'w', 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    export_fig(fullfile(output_directory, sprintf('echo%d_phase', idx)), '-r300', '-tif'); % '-c[100,680,210,870]' [top,right,bottom,left]
    close gcf;
end

%% Display all slices as a montage
%--------------------------------------------------------------------------
% fB0
%--------------------------------------------------------------------------
c = floor(Ns/2) + 1;
B0map_montage = zeros(N1*2, N2*c, 'double');
B0map_montage(1:N1,(1:N2*c)) = reshape(reorient(B0map_nlinv(:,:,1:c)), [N1 N2*c]);
B0map_montage((1:N1)+N1,1:N2*(c-1)) = reshape(reorient(B0map_nlinv(:,:,c+1:end)), [N1 N2*(Ns-c)]);

figure('Color', 'k', 'Position', [1 1 1600 823]);
imagesc(B0map_montage); axis image;
set(gca, 'XColor', 'w', 'YColor', 'w', 'XTick', [], 'YTick', []);
caxis(hz_range);
colormap(hot(256));
hc = colorbar;
set(hc, 'Color', 'w', 'FontSize', 14);
text(N2 * c / 2, 0, {sprintf('NLINV estimation (\\Deltaf) [Hz], \\alpha_{min} = %3.1e, iter = %d', alpha_min, irgnm_iterations)}, 'Color', 'w', 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
export_fig(fullfile(output_directory, sprintf('B0map_nlinv_ro%d_min%3.1e_w%d_iter%d', remove_oversampling, alpha_min, w_fB0, irgnm_iterations)), '-r300', '-tif'); % '-c[100,680,210,870]' [top,right,bottom,left]
close gcf;

%--------------------------------------------------------------------------
% rho
%--------------------------------------------------------------------------
rho_montage = zeros(N1*2, N2*c, 'double');
rho_montage(1:N1,(1:N2*c)) = reshape(reorient(rho_nlinv(:,:,1:c)), [N1 N2*c]);
rho_montage((1:N1)+N1,1:N2*(c-1)) = reshape(reorient(rho_nlinv(:,:,c+1:end)), [N1 N2*(Ns-c)]);

figure('Color', 'k', 'Position', [1 1 1600 823]);
imagesc(abs(rho_montage)); axis image;
set(gca, 'XColor', 'w', 'YColor', 'w', 'XTick', [], 'YTick', []);
colormap(gray(256));
hc = colorbar;
set(hc, 'Color', 'w', 'FontSize', 14);
text(N2 * c / 2, 0, {sprintf('Magnitude of NLINV estimation (\\rho), \\alpha_{min} = %3.1e, iter = %d', alpha_min, irgnm_iterations)}, 'Color', 'w', 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
export_fig(fullfile(output_directory, sprintf('rho_nlinv_mag_ro%d_min%3.1e_w%d_iter%d', remove_oversampling, alpha_min, w_fB0, irgnm_iterations)), '-r300', '-tif'); % '-c[100,680,210,870]' [top,right,bottom,left]
close gcf;

figure('Color', 'k', 'Position', [1 1 1600 823]);
imagesc(angle(rho_montage)*180/pi); axis image;
set(gca, 'XColor', 'w', 'YColor', 'w', 'XTick', [], 'YTick', []);
colormap(hsv(256));
hc = colorbar;
set(hc, 'Color', 'w', 'FontSize', 14);
text(N2 * c / 2, 0, {sprintf('Phase of NLINV estimation (\\rho), \\alpha_{min} = %3.1e, iter = %d', alpha_min, irgnm_iterations)}, 'Color', 'w', 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
export_fig(fullfile(output_directory, sprintf('rho_nlinv_phase_ro%d_min%3.1e_w%d_iter%d', remove_oversampling, alpha_min, w_fB0, irgnm_iterations)), '-r300', '-tif'); % '-c[100,680,210,870]' [top,right,bottom,left]
close gcf;

%% Calculate a mask defining the image and noise regions using the iterative intermean algorithm
%--------------------------------------------------------------------------
% Calculate the maximum magnitude of images from all echo points
%--------------------------------------------------------------------------
im_max = abs(rho_nlinv);

%--------------------------------------------------------------------------
% Calculate a mask
%--------------------------------------------------------------------------
mask_support = false(N1, N2, Ns);
for idx = 1:Ns
    level = isodata(im_max(:,:,idx), 'log');
    mask_support(:,:,idx) = (im_max(:,:,idx) > level);
end

%--------------------------------------------------------------------------
% Fill voids
%--------------------------------------------------------------------------
mask_support = bwareaopen(mask_support, 30, 4);
mask_support = imfill(mask_support, 'holes');

%--------------------------------------------------------------------------
% Dilate the mask
%--------------------------------------------------------------------------
se = strel('disk', 8);
mask_support = imdilate(mask_support, se);

%--------------------------------------------------------------------------
% Fill holes
%--------------------------------------------------------------------------
mask_support = imfill(mask_support, 'holes');

%% Calculate a masked static off-resonance map [Hz]
B0map_nlinv_mask = B0map_nlinv .* mask_support;

%% Display all slices as a montage (with a mask)
%--------------------------------------------------------------------------
% fB0
%--------------------------------------------------------------------------
B0map_montage_mask = zeros(N1*2, N2*c, 'double');
B0map_montage_mask(1:N1,(1:N2*c)) = reshape(reorient(B0map_nlinv_mask(:,:,1:c)), [N1 N2*c]);
B0map_montage_mask((1:N1)+N1,1:N2*(c-1)) = reshape(reorient(B0map_nlinv_mask(:,:,c+1:end)), [N1 N2*(Ns-c)]);

figure('Color', 'k', 'Position', [1 1 1600 823]);
imagesc(B0map_montage_mask); axis image;
set(gca, 'XColor', 'w', 'YColor', 'w', 'XTick', [], 'YTick', []);
caxis(hz_range);
colormap(hot(256));
hc = colorbar;
set(hc, 'Color', 'w', 'FontSize', 14);
text(N2 * c / 2, 0, {sprintf('NLINV estimation (\\Deltaf) [Hz], \\alpha_{min} = %3.1e, iter = %d', alpha_min, irgnm_iterations)}, 'Color', 'w', 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
export_fig(fullfile(output_directory, sprintf('B0map_nlinv_ro%d_min%3.1e_w%d_iter%d_mask', remove_oversampling, alpha_min, w_fB0, irgnm_iterations)), '-r300', '-tif'); %  '-c[100,680,210,870]' [top,right,bottom,left]
close gcf;

%% Save the estimated B0 maps
save(fullfile(output_directory, sprintf('B0map_nlinv_ro%d_min%3.1e_w%d_iter%d.mat', remove_oversampling, alpha_min, w_fB0, irgnm_iterations)), 'B0map_nlinv', 'rho_nlinv', 'B0map_nlinv_mask', '-v7.3');
