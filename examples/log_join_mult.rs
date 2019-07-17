#[cfg(feature = "logs")]
extern crate rayon_logs as rayon;
use ndarray::Array;
use rayon::subgraph;
use rayon::ThreadPoolBuilder;
use rayon_adaptive::prelude::*;
use rayon_adaptive::Policy;

use matrix_mult::matrix;
use matrix_mult::matrix_adaptive;
use ndarray::linalg;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let height = 4000;
    let width = 4000;
    let an = Array::from_shape_fn((height, width), |(i, j)| {
        ((((j + i * width) % 3) as f32 + random) as f32)
    });
    let bn = Array::from_shape_fn((width, height), |(i, j)| {
        ((((j + 7 + i * height) % 3) as f32 - random) as f32)
    });
    let mut dest = Array::zeros((height, height));
    // let mut vec = Vec::new();
    // vec.push(([(an.view(),bn.view())].to_vec(),dest.view_mut()));
    // let mat = matrix::Matrix {
    //     matrix: vec,
    // };

    let matrix_half = matrix_adaptive::Matrix {
        a: an.view(),
        b: bn.view(),
        d: dest.view_mut(),
    };

    let pool = ThreadPoolBuilder::new()
        .build()
        .expect("Pool creation failed");


    pool.logging_install(|| {
        matrix_half
            .cut()
            .with_policy(Policy::Join(4000 * 4000 / 64 + 1))
            .for_each(|e| {
                let (r, c) = e.d.dim();
                subgraph("prod", (r * c) ^ 3, || {
                    let (a, b, mut d) = (e.a, e.b, e.d);
                    linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut d);
                })
            })
    })
    .1
    .save_svg("join_64_8threads.svg")
    .expect("Saving failed");
}
