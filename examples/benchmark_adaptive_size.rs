extern crate rayon;
use itertools::enumerate;
use matrix_mult::benchmark;
use ndarray::LinalgScalar;
use ndarray::{Array, ArrayView, Ix2,linalg};
use rayon_adaptive::Policy;
use std::fs::File;
use std::io::Write;
use matrix_mult::vectorisation_packed_simd;
use rand::Rng;

const ITERS: usize = 5;

fn average(numbers: [f64; ITERS as usize]) -> f64 {
    numbers.iter().sum::<f64>() / numbers.len() as f64
}

fn main() {
    do_benchmark("OPENBLAS_new_cut_adaptive_size_test.data", |s,a,b,p| benchmark::benchmark_adaptive_generic(s,a,b, 
    p, |a,b,mut c| linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut c)));
}

pub fn do_benchmark<
    F: Fn(usize, ArrayView<f32, Ix2>, ArrayView<f32, Ix2>, Policy) -> u64,
>(
    filename: &str,
    f: F,
) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    let input_size: Vec<usize> =  (1..41).map(|i| i * 250).collect();

    for (_j, size) in enumerate(input_size) {
        let mut adaptive16 = [0f64; ITERS as usize];
        let mut adaptive32 = [0f64; ITERS as usize];
        let mut adaptive48 = [0f64; ITERS as usize];
        let mut adaptive64 = [0f64; ITERS as usize];
        let mut adaptive80 = [0f64; ITERS as usize];
        let mut adaptive96 = [0f64; ITERS as usize];
        let mut adaptive112 = [0f64; ITERS as usize];
        let mut adaptive128 = [0f64; ITERS as usize];
   
        eprintln!("SIZE {:?}", size);
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
            let mut size_iter = (1..9).map(|i| i as f32 * 16.0);
            adaptive16[i] = f(height, an.view(), bn.view(), Policy::Adaptive(f32::log2((height*height) as f32) as usize ,(size_iter.next().unwrap()*f32::sqrt((height*height) as f32)) as usize)) as f64;
            adaptive32[i] = f(height, an.view(), bn.view(), Policy::Adaptive(f32::log2((height*height) as f32) as usize ,(size_iter.next().unwrap()*f32::sqrt((height*height) as f32)) as usize)) as f64;
            adaptive48[i] = f(height, an.view(), bn.view(), Policy::Adaptive(f32::log2((height*height) as f32) as usize ,(size_iter.next().unwrap()*f32::sqrt((height*height) as f32)) as usize)) as f64;
            adaptive64[i] = f(height, an.view(), bn.view(), Policy::Adaptive(f32::log2((height*height) as f32) as usize ,(size_iter.next().unwrap()*f32::sqrt((height*height) as f32)) as usize)) as f64;
            adaptive80[i] = f(height, an.view(), bn.view(), Policy::Adaptive(f32::log2((height*height) as f32) as usize ,(size_iter.next().unwrap()*f32::sqrt((height*height) as f32)) as usize)) as f64;
            adaptive96[i] = f(height, an.view(), bn.view(), Policy::Adaptive(f32::log2((height*height) as f32) as usize ,(size_iter.next().unwrap()*f32::sqrt((height*height) as f32)) as usize)) as f64;
            adaptive112[i] = f(height, an.view(), bn.view(), Policy::Adaptive(f32::log2((height*height) as f32) as usize ,(size_iter.next().unwrap()*f32::sqrt((height*height) as f32)) as usize)) as f64;
            adaptive128[i] = f(height, an.view(), bn.view(), Policy::Adaptive(f32::log2((height*height) as f32) as usize ,(size_iter.next().unwrap()*f32::sqrt((height*height) as f32)) as usize)) as f64;
        }

        file.write_all(
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                size,
                average(adaptive16),
                average(adaptive32),
                average(adaptive48),
                average(adaptive64),
                average(adaptive80),
                average(adaptive96),
                average(adaptive112),
                average(adaptive128),
            )
            .as_bytes(),
        )?;
    }
    Ok(())
}
