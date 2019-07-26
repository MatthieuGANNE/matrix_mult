use crate::matrix;
use crate::matrix_adaptive;
use ndarray::linalg;
use ndarray::{Array, ArrayView, Ix2,ArrayViewMut};
use rayon_adaptive::prelude::*;
use rayon_adaptive::Policy;

///
/// A(x,y) * B(y,x)
/// Return the time of a matrix multiplication
/// The matrix result must be a square matrix
/// The strategy is the Policy (Adaptive,Join,Rayon,...)
/// The resolution must be a sequential algorithm of matrix multiplication
/// This function use a matrix representation that cut in 2 peaces (power 2 friendly) along the larger axis at each call divide_at
/// 
pub fn benchmark_adaptive_generic<F>(
    height: usize,
    a: ArrayView<f32, Ix2>,
    b: ArrayView<f32, Ix2>,
    strategy: Policy,
    resolution: F,
) -> u64 
where 
F: Fn(ArrayView<f32, Ix2>, ArrayView<f32, Ix2>,ArrayViewMut<f32,Ix2>) + Copy + Sync
{
    let (ar,ac) = a.dim();
    let (br,bc) = b.dim();
    assert_eq!(ac,br);
    assert_eq!(height,bc);
    assert_eq!(height,ar);
    let mut dest = Array::zeros((height, height));
    let (destdim1,destdim2) = dest.dim();
    let mat = matrix_adaptive::Matrix {
        a: a,
        b: b,
        d: dest.view_mut(),
        asize: a.dim(),
        bsize: b.dim(),
        dsize: (destdim1,destdim2),
    };
    let start_time = time::precise_time_ns();
    mat.with_policy(strategy).for_each(|e| {
        let (a, b, d) = (e.a, e.b, e.d);
        let dima = a.shape();
        let dimb = b.shape();
        let dimd = d.shape();
                if !(dima[0] == 0
                    || dima[1] == 0
                    || dimb[0] == 0
                    || dimb[1] == 0
                    || dimd[0] == 0
                    || dimd[1] == 0)
                {
        resolution(a,b,d);
                }
    });
    let end_time = time::precise_time_ns();
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1f32, &a, &b, 1f32, &mut verif);
    assert_abs_diff_eq!(
        dest.as_slice().unwrap(),
        verif.as_slice().unwrap(),
        epsilon = 5e-1f32
    );
    (end_time - start_time)
}

///
/// A(x,y) * B(y,x)
/// Return the time of a matrix multiplication
/// The matrix result must be a square matrix
/// The strategy is the Policy (Join,Rayon,...)
/// The resolution must be a sequential algorithm of matrix multiplication
/// This function use a matrix representation that cut in 4 peaces (power 2 friendly) at each call divide_at
/// 
pub fn benchmark_basic_generic<F>(
    height: usize,
    a: ArrayView<f32, Ix2>,
    b: ArrayView<f32, Ix2>,
    strategy: Policy,
    resolution: F,
) -> u64 
where 
F: Fn(ArrayView<f32, Ix2>, ArrayView<f32, Ix2>,&mut ArrayViewMut<f32,Ix2>) + Copy + Sync
{
    let mut dest = Array::zeros((height, height));
    let mut vec = Vec::new();
    vec.push(([(a, b)].to_vec(), dest.view_mut()));
    let mat = matrix::Matrix { matrix: vec };
    let start_time = time::precise_time_ns();
    mat.cut().with_policy(strategy).for_each(|e| {
        for (vect, mut output) in e.matrix {
            for (a, b) in vect {
                let dima = a.shape();
                let dimb = b.shape();
                let dimd = output.shape();
                if dima[0] == 0
                    || dima[1] == 0
                    || dimb[0] == 0
                    || dimb[1] == 0
                    || dimd[0] == 0
                    || dimd[1] == 0
                {
                    continue;
                }
                resolution(a,b,&mut output);
            }
        }
    });
    let end_time = time::precise_time_ns();

    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut verif);
    assert_abs_diff_eq!(
        dest.as_slice().unwrap(),
        verif.as_slice().unwrap(),
        epsilon = 5e-1f32
    );
    (end_time - start_time)
}