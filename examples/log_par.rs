#[cfg(feature = "logs")]
extern crate rayon_logs as rayon;
use ndarray::Array;
use rayon::subgraph;
use rayon::ThreadPoolBuilder;
use rayon_adaptive::prelude::*;
use rayon_adaptive::Policy;

use matrix_mult::matrix;
use matrix_mult::matrix_adaptive;
use ndarray::{linalg,ArrayView,ArrayViewMut};
use rand::Rng;
use matrix_mult::faster_vec;
use matrix_mult::naive_sequential;
use matrix_mult::vectorisation_packed_simd;
fn main() {
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let height = 500;
    let width = 500;
 
  
    let pool = ThreadPoolBuilder::new()
        .build()
        .expect("Pool creation failed");

    pool.compare()
        .attach_algorithm_with_setup("vectorisation Join", || {
            let an = Array::from_shape_fn((height, width), |(i, j)| {
            ((((j + i * width) % 3) as f32 + random) as f32)
            });
            let bn = Array::from_shape_fn((width, height), |(i, j)| {
            ((((j + 7 + i * height) % 3) as f32 - random) as f32)
            });
            let mut dest = Array::zeros((height, height));
            (an,bn,dest)
            } , |(a,b,mut c)| {
                let (ddim1,ddim2)= c.dim();
                let mat = matrix_adaptive::Matrix {
                    a: a.view(),
                    b: b.view(),
                    d: c.view_mut(),
                    asize: a.dim(),
                    bsize: b.dim(),
                    dsize: (ddim1,ddim2),
                };
                mat.cut().with_policy(Policy::Join(height * height / 64)).for_each(|e| {
                    let (a, b, mut d) = (e.a, e.b, e.d);
                    let (row,col) = d.dim();
                    subgraph("work_op", (row * row * row), || {
                        vectorisation_packed_simd::mult_faster_from_ndarray(a,b,&mut d);
                        })
                    })
                })
        .attach_algorithm_with_setup("vectorisation seq blocks ", || {
            let an = Array::from_shape_fn((height, width), |(i, j)| {
            ((((j + i * width) % 3) as f32 + random) as f32)
            });
            let bn = Array::from_shape_fn((width, height), |(i, j)| {
            ((((j + 7 + i * height) % 3) as f32 - random) as f32)
            });
            let mut dest = Array::zeros((height, height));
            (an,bn,dest)
            } , |(a,b,mut c)|{
                let (row,col) = c.dim();
                let (avec, bvec, rvec) = naive_sequential::cut_in_blocks(a.view(), b.view(), c.view_mut(), 600, 600);
                    naive_sequential::mult_blocks(avec, bvec, rvec, |a,b,mut c| {
                        let (row,col) = c.dim();
                        subgraph("work_op", row * row * row, || vectorisation_packed_simd::mult_faster_from_ndarray(a,b,&mut c))
                    });
        })
    .generate_logs("vectorisation_par.html")
    .expect("writing logs failed");
}
