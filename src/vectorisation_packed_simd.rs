use packed_simd::f32x8;
use ndarray::{ArrayView,ArrayViewMut,Ix2};
use crate::my_ndarray;
use crate::vectorisation;
use std::slice::{from_raw_parts, from_raw_parts_mut};
#[cfg(test)]
use crate::naive_sequential;
#[cfg(test)]
use ndarray::{linalg,Array};
#[cfg(test)]
use rand::Rng;

fn multiply_add_packed_sim(  mut into: &mut [f32],
    a: f32,
    b: &[f32],
    awidth: usize,
    aheight: usize,
    bwidth: usize,
    bheight: usize,
    intowidth: usize,
    intoheight: usize,){

    let achunk = f32x8::splat(a);
    b.chunks_exact(8)
    .zip(into.chunks_exact_mut(8))
    .for_each(|(x,y)| { 
        let chunkx = f32x8::from_slice_unaligned(x);
        let chunky = f32x8::from_slice_unaligned(y);
        let res = chunkx.mul_add(achunk ,chunky);
        res.write_to_slice_unaligned(y);
    });
    let len = b.len();
    let calc_len = len - len%8;
    b[calc_len..len].iter().zip(into[calc_len..len].iter_mut())
    .for_each(|(x,y)|{
        *y = a*x+*y

    });
}

pub fn mult_faster_from_ndarray(a: ArrayView<f32,Ix2> ,b: ArrayView<f32,Ix2>,output: &mut ArrayViewMut<f32,Ix2>) {
    let (raw_ptr_a, len_a) = my_ndarray::view_ptr(a);
    let stridesa = a.strides();
    let (raw_ptr_b, len_b) = my_ndarray::view_ptr(b);
    let stridesb = b.strides();
    let raw_ptr_r = output.as_mut_ptr();
    let dimr = output.shape();
    let dima = a.shape();
    let dimb = b.shape();
    let (row, col) = (dimr[0], dimr[1]);
    let strides = output.strides();
    let len_r = (row - 1) * strides[0] as usize + col;
    let slicea = unsafe { from_raw_parts(raw_ptr_a, len_a) };
    let sliceb = unsafe { from_raw_parts(raw_ptr_b, len_b) };
    let mut slicer = unsafe { from_raw_parts_mut(raw_ptr_r, len_r) };
    vectorisation::multiply_add(
                    &mut slicer,
                    &slicea,
                    &sliceb,
                    dima[1],
                    dima[0],
                    dimb[1],
                    dimb[0],
                    dimr[1],
                    dimr[0],
                    stridesa[0] as usize,
                    stridesb[0] as usize,
                    strides[0] as usize,
                    multiply_add_packed_sim,
                );
}

#[test]
fn test_mult_blocked() {
    let height = 1000;
    let width = 1000;
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let an = Array::from_shape_fn((height, width), |(i, j)| {
        (((j + i * width) % 3) as f32) + random
    });
    let bn = Array::from_shape_fn((width, height), |(i, j)| {
        (((j + 7 + i * height) % 3) as f32) - random
    });
    let mut dn = Array::zeros((height, height));
    let (avec, bvec, rvec) = naive_sequential::cut_in_blocks(an.view(), bn.view(), dn.view_mut(),300,300);
    naive_sequential::mult_blocks(avec, bvec, rvec,|a,b,mut c| mult_faster_from_ndarray(a, b, &mut c));
            
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);

    assert_abs_diff_eq!(
        dn.as_slice().unwrap(),
        verif.as_slice().unwrap(),
        epsilon = 1e-1f32
    );
}

#[test]
fn test_mult_faster() {
    let height = 1000;
    let width = 1000;
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let an = Array::from_shape_fn((height, width), |(i, j)| {
        (((j + i * width) % 3) as f32) + random
    });
    let bn = Array::from_shape_fn((width, height), |(i, j)| {
        (((j + 7 + i * height) % 3) as f32) - random
    });
    let mut dn = Array::zeros((height, height));
    mult_faster_from_ndarray(an.view(), bn.view(), &mut dn.view_mut());
            
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);

    assert_abs_diff_eq!(
        dn.as_slice().unwrap(),
        verif.as_slice().unwrap(),
        epsilon = 1e-1f32
    );
}
