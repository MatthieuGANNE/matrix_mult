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
    do_benchmark("OPENBLAS_new_cut_adaptive_size.data", |s,a,b,p| benchmark::benchmark_adaptive_generic(s,a,b, 
    p, |a,b,mut c| linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut c)));
    // do_benchmark("benchmark_f32_vectorisation_simd_new_cut.data", |s,a,b,p| benchmark::benchmark_adaptive_generic(s,a,b, 
    // p, |a,b,mut c| vectorisation_packed_simd::mult_faster_from_ndarray(a,b,&mut c)));
    // do_benchmark("benchmark_f32_vectorisation_simd_basic.data", |s,a,b,p| benchmark::benchmark_basic_generic(s,a,b, 
    // p, |a,b,mut c| vectorisation_packed_simd::mult_faster_from_ndarray(a,b,&mut c)));
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
        let mut rayon1 = [0f64; ITERS as usize];
        let mut rayon100 = [0f64; ITERS as usize];
        let mut sequential = [0f64; ITERS as usize];
        let mut join = [0f64; ITERS as usize];
        let mut adaptive = [0f64; ITERS as usize];
        let mut speeduprayon1 = [0f64; ITERS as usize];
        let mut speeduprayon1000 = [0f64; ITERS as usize];
        let mut speedupjoin = [0f64; ITERS as usize];
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

            // Rayon 1
            // rayon1[i] = f(height, an.view(), bn.view(), Policy::Rayon(1)) as f64;
            // // Rayon 1000
            // rayon100[i] = f(height, an.view(), bn.view(), Policy::Rayon(1_000)) as f64;
            // Sequential
            // sequential[i] = f(height, an.view(), bn.view(), Policy::Sequential) as f64;
            // Join
            // join[i] = (f(
            //     height,
            //     an.view(),
            //     bn.view(),
            //     Policy::Join(size * size / 64 + 1),
            // )) as f64;

            adaptive[i] = f(height, an.view(), bn.view(), Policy::Adaptive(f32::log2((height*height) as f32) as usize ,(f32::sqrt((height*height) as f32)) as usize)) as f64;
            // speedup
            speeduprayon1[i] = sequential[i] / rayon1[i];
            speeduprayon1000[i] = sequential[i] / rayon100[i];
            speedupjoin[i] = sequential[i] / join[i];
        }

        file.write_all(
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                size,
                average(rayon1),
                average(rayon100),
                average(sequential),
                average(join),
                average(adaptive),
                average(speeduprayon1),
                average(speeduprayon1000),
                average(speedupjoin),
            )
            .as_bytes(),
        )?;
    }
    Ok(())
}
