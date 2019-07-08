use crate::faster_vec;
use crate::matrix;
use crate::matrix_adaptive;
use crate::my_ndarray;
use ndarray::linalg;
use ndarray::{Array, ArrayView, Ix2};
use rayon_adaptive::prelude::*;
use rayon_adaptive::Policy;
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub fn benchmark(
    height: usize,
    a: ArrayView<f32, Ix2>,
    b: ArrayView<f32, Ix2>,
    strategy: Policy,
) -> u64 {
    let mut dest = Array::zeros((height, height));
    let mut vec = Vec::new();
    vec.push(([(a, b)].to_vec(), dest.view_mut()));
    let mat = matrix::Matrix { matrix: vec };
    let start_time = time::precise_time_ns();
    mat.cut().with_policy(strategy).for_each(|e| {
        for (vect, mut output) in e.matrix {
            for (a, b) in vect {
                linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut output);
            }
        }
    });
    let end_time = time::precise_time_ns();

    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut verif);
    assert_eq!(verif, dest);
    (end_time - start_time)
}

pub fn benchmark_faster(
    height: usize,
    a: ArrayView<f32, Ix2>,
    b: ArrayView<f32, Ix2>,
    strategy: Policy,
) -> u64 {
    let mut dest = Array::zeros((height, height));
    let mut vec = Vec::new();
    vec.push(([(a, b)].to_vec(), dest.view_mut()));
    let mat = matrix::Matrix { matrix: vec };
    let start_time = time::precise_time_ns();
    mat.cut().with_policy(strategy).for_each(|e| {
        for (vect, mut output) in e.matrix {
            let (rrow, rcol) = output.dim();
            for (a, b) in vect {
                let dima = a.shape();
                let dimb = b.shape();
                if dima[0] == 0
                    || dima[1] == 0
                    || dimb[0] == 0
                    || dimb[1] == 0
                    || rrow == 0
                    || rcol == 0
                {
                    continue;
                }
                let (raw_ptr_a, len_a) = my_ndarray::view_ptr(a);
                let stridesa = a.strides();
                let (raw_ptr_b, len_b) = my_ndarray::view_ptr(b);
                let stridesb = b.strides();
                let raw_ptr_r = output.as_mut_ptr();
                let dimr = output.shape();
                let (row, col) = (dimr[0], dimr[1]);
                let strides = output.strides();
                let len_r = (row - 1) * strides[0] as usize + col;
                let slicea = unsafe { from_raw_parts(raw_ptr_a, len_a) };
                let sliceb = unsafe { from_raw_parts(raw_ptr_b, len_b) };
                let mut slicer = unsafe { from_raw_parts_mut(raw_ptr_r, len_r) };
                faster_vec::multiply_add(
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
                );
            }
        }
    });
    let end_time = time::precise_time_ns();
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut verif);
    assert_eq!(verif, dest);
    (end_time - start_time)
}

pub fn benchmark_adaptive(
    height: usize,
    a: ArrayView<f32, Ix2>,
    b: ArrayView<f32, Ix2>,
    strategy: Policy,
) -> u64 {
    let mut dest = Array::zeros((height, height));
    let mat = matrix_adaptive::Matrix {
        a: a,
        b: b,
        d: dest.view_mut(),
    };
    let start_time = time::precise_time_ns();
    mat.cut().with_policy(strategy).for_each(|e| {
        let (a, b, mut d) = (e.a, e.b, e.d);
        linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut d);
    });
    let end_time = time::precise_time_ns();
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut verif);
    assert_eq!(verif, dest);
    (end_time - start_time)
}
