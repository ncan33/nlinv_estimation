# nlinv_estimation
 
Nonlinear inversion (**NLINV**)-based parameter estimation (e.g., water and static off-resonance) in the image domain.

The theory is described in

**"MaxGIRF: Image Reconstruction Incorporating Concomitant
Field and Gradient Impulse Response Function Effects"**, by Nam G. Lee, Rajiv Ramasawmy, Yongwan Lim, Adrienne E. Campbell-Washburn, and Krishna S. Nayak.

This code is distributed under the BSD license.

Nam Gyun Lee, University of Southern California, Dec 2021.

## Example usage
   
Run `demo_cartesian_recon_GRE_datasets.m` from [this repository](https://github.com/usc-mrel/lowfield_maxgirf).

Update `multiecho_fullpath` in `demo_human_nlinv_estimation_rho_B0.m`.

Run `demo_human_nlinv_estimation_rho_B0.m` when only CPUs are available or `demo_human_nlinv_estimation_rho_B0_gpu.m` if a single GPU is available.
