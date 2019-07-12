use crate::faster_vec;
use crate::matrix;
use crate::matrix_adaptive;
use ndarray::linalg;
use ndarray::{Array, ArrayView, Ix2};
use rayon_adaptive::prelude::*;
use rayon_adaptive::Policy;

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
                faster_vec::mult_faster_from_ndarray(a,b,&mut output);
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
