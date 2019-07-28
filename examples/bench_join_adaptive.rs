#[macro_use]
extern crate approx;
use itertools::enumerate;
use ndarray::linalg;
use ndarray::{LinalgScalar,ArrayView,ArrayViewMut,Array, Ix2};
use rand::Rng;
use std::fs::File;
use std::io::Write;
use matrix_mult::naive_sequential;
use matrix_mult::faster_vec;
use std::ops::AddAssign;
use matrix_mult::vectorisation;
use matrix_mult::matrix_adaptive;
use rayon_adaptive::prelude::*;
use rayon_adaptive::Policy;
const ITERS: usize = 5;

fn average(numbers: [f64; ITERS as usize]) -> f64 {
    numbers.iter().sum::<f64>() / numbers.len() as f64
}

fn main() -> std::io::Result<()> {
    let input_size: Vec<usize> = (1..41).map(|i| i * 250).collect();
    do_benchmark("openblas_join_adaptive.data", |a,b,mut c|  linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut c),input_size);
    Ok(())
}   

fn do_benchmark<F>(filename: &str,resolution : F, input_size: Vec<usize>) -> std::io::Result<()> 
where
F: Fn(ArrayView<f32, Ix2>, ArrayView<f32, Ix2>, ArrayViewMut<f32,Ix2>) + Copy + Sync
{
    let mut file = File::create(filename)?;  
    for (_j, size) in enumerate(input_size) {
        let mut time= [0f64; ITERS as usize];
        println!("SIZE : {:?}", size);
            for i in 0..ITERS {
            let height = size as usize;
            let width = size as usize;
            let mut rng = rand::thread_rng();
            let random = rng.gen_range(0.0, 1.0);
            let an = Array::from_shape_fn((height, width), |(i, j)| {
                ((j + i * width) % 3) as f32 + random 
            });
            let bn = Array::from_shape_fn((width, height), |(i, j)| {
                ((j + i * height) % 3) as f32 - random
            });
            let mut dest = Array::zeros((height, height));
                let (ddim1,ddim2)= dest.dim();
            let matrix_half = matrix_adaptive::Matrix {
                a: an.view(),
                b: bn.view(),
                d: dest.view_mut(),
                asize: an.dim(),
                bsize: bn.dim(),
                dsize: (ddim1,ddim2),
            };

            let start_time = time::precise_time_ns();
             matrix_half
            .with_policy(Policy::Join(height*height/64))
            .for_each(|mut e| {
                let (ra,ca) = e.a.dim();
                let (rb,cb) = e.b.dim();
                let (rd,cd) = e.d.dim();
                e.asize = (ra,ca);
                e.bsize = (rb,cb);
                e.dsize = (rd,cd);
                e.with_policy(Policy::Adaptive((8.0*f32::log2((rd*cd) as f32)) as usize ,
                    ((16.0*f32::sqrt((rd*cd) as f32) as f32) as usize ))).for_each( |e| {
                    resolution(e.a,e.b,e.d);
                })
            });
            let end_time = time::precise_time_ns();

            let mut verif = Array::zeros((height, height));
            linalg::general_mat_mul(1f32, &an.view(), &bn.view(), 1f32, &mut verif.view_mut());
            assert_abs_diff_eq!(
                dest.as_slice().unwrap(),
                verif.as_slice().unwrap(),
                epsilon = 1e-1f32
            );
            time[i] = (end_time - start_time) as f64;
        }
        file.write_all(format!("{}\t{}\n", size, average(time)).as_bytes())?;
    }
    Ok(())
}