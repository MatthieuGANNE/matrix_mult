extern crate rayon;
use itertools::enumerate;
use matrix_mult::benchmark;
use ndarray::LinalgScalar;
use ndarray::{Array, ArrayView, Ix2};
use rayon_adaptive::Policy;
use std::fs::File;
use std::io::Write;
const ITERS: usize = 10;

fn average(numbers: [f64; ITERS as usize]) -> f64 {
    numbers.iter().sum::<f64>() / numbers.len() as f64
}

fn main() {
    // do_benchmark("benchmark_f32_ndarray.data_4C", benchmark::benchmark);
    do_benchmark("benchmark_f32_faster_seq_naive", benchmark::benchmark_faster);
    // do_benchmark(
    //     "benchmark_f32_adaptive.data_4C",
    //     benchmark::benchmark_adaptive,
    // );
}

pub fn do_benchmark<
    T: From<u16> + LinalgScalar,
    F: Fn(usize, ArrayView<T, Ix2>, ArrayView<T, Ix2>, Policy) -> u64,
>(
    filename: &str,
    f: F,
) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    let input_size: Vec<usize> =  (1..15).map(|i| i * 200).collect();

    for (_j, size) in enumerate(input_size) {
        let mut rayon1 = [0f64; ITERS as usize];
        let mut rayon100 = [0f64; ITERS as usize];
        let mut sequential = [0f64; ITERS as usize];
        let mut join = [0f64; ITERS as usize];
        let mut speeduprayon1 = [0f64; ITERS as usize];
        let mut speeduprayon1000 = [0f64; ITERS as usize];
        let mut speedupjoin = [0f64; ITERS as usize];
        eprintln!("Seq {:?}", size);
        for i in 0..ITERS {
            let height = size as usize;
            let width = size as usize;
            let an = Array::from_shape_fn((height, width), |(i, j)| {
                T::from(((j + i * width) % 3) as u16)
            });
            let bn = Array::from_shape_fn((width, height), |(i, j)| {
                T::from(((j + i * height) % 3) as u16)
            });

            // Rayon 1
            // rayon1[i] = f(height, an.view(), bn.view(), Policy::Rayon(1)) as f64;
            // eprintln!("RAYON");
            // Rayon 1000
            // rayon100[i] = f(height, an.view(), bn.view(), Policy::Rayon(1_000)) as f64;
            // Sequential
            sequential[i] = f(height, an.view(), bn.view(), Policy::Sequential) as f64;
            // eprintln!("JOIN {:?}", size);
            // // Join
            // join[i] = (f(
            //     height,
            //     an.view(),
            //     bn.view(),
            //     Policy::Join(size * size / 64 + 1),
            // )) as f64;

            // speedup
            // speeduprayon1[i] = sequential[i] / rayon1[i];
            // speeduprayon1000[i] = sequential[i] / rayon100[i];
            // speedupjoin[i] = sequential[i] / join[i];
        }

        file.write_all(
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                size,
                average(rayon1),
                average(rayon100),
                average(sequential),
                average(join),
                average(speeduprayon1),
                average(speeduprayon1000),
                average(speedupjoin),
            )
            .as_bytes(),
        )?;
    }
    Ok(())
}
