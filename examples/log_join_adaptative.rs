#[cfg(feature = "logs")]
extern crate rayon_logs as rayon;
use ndarray::Array;
use rayon::subgraph;
use rayon::ThreadPoolBuilder;
use rayon_adaptive::prelude::*;
use rayon_adaptive::Policy;

use matrix_mult::matrix;
use matrix_mult::vectorisation_packed_simd;
use matrix_mult::faster_vec; 
use matrix_mult::matrix_adaptive;
use ndarray::linalg;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let height = 3000;
    let width = 3000;
    let an = Array::from_shape_fn((height, width), |(i, j)| {
        ((((j + i * width) % 3) as f32 + random) as f32)
    });
    let bn = Array::from_shape_fn((width, height), |(i, j)| {
        ((((j + 7 + i * height) % 3) as f32 - random) as f32)
    });
    let mut dest = Array::zeros((height, height));
    let (ddim1,ddim2)= dest.dim();
    // let mut vec = Vec::new();
    // vec.push(([(an.view(),bn.view())].to_vec(),dest.view_mut()));
    // let mat = matrix::Matrix {
    //     matrix: vec,
    // };

    let matrix_half = matrix_adaptive::Matrix {
        a: an.view(),
        b: bn.view(),
        d: dest.view_mut(),
        asize: an.dim(),
        bsize: bn.dim(),
        dsize: (ddim1,ddim2),
    };

    let pool = ThreadPoolBuilder::new()
        .build()
        .expect("Pool creation failed");


    pool.logging_install(|| {
        matrix_half
            .with_policy(Policy::Join(3000*3000/64))
            .for_each(|mut e| {
                let (ra,ca) = e.a.dim();
                let (rb,cb) = e.b.dim();
                let (rd,cd) = e.d.dim();
                e.asize = (ra,ca);
                e.bsize = (rb,cb);
                e.dsize = (rd,cd);
                e.with_policy(Policy::Adaptive(30,3000)).for_each( |e| {
                    let (ra,ca) = e.a.dim();
                    let (rd,cd) = e.d.dim();
                    subgraph("prod", ra * ca * cd , || {
                        let (a, b, mut d) = (e.a, e.b, e.d);
                        linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut d)
                    })
                    //vectorisation_packed_simd::mult_faster_from_ndarray(a,b,&mut d);
                })
            })
    })
    .1
    .save_svg("test_Join_Adaptive.svg")
    .expect("Saving failed");
}
