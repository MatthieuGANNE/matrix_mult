extern crate rayon;
use itertools::enumerate;
use ndarray::linalg;
use ndarray::Array;
use rand::Rng;
use std::fs::File;
use std::io::Write;

const ITERS: usize = 20;

fn average(numbers: [f64; ITERS as usize]) -> f64 {
    numbers.iter().sum::<f64>() / numbers.len() as f64
}

fn main() -> std::io::Result<()> {
    let mut file = File::create("algomult_seq_blas_debut.data")?;

    let input_size: Vec<usize> = (1..250).map(|i| i * 20).collect();

    for (_j, size) in enumerate(input_size) {
        println!("SIZE : {:?}", size);
        let mut timeblas = [0f64; ITERS as usize];
        let height = size as usize;
        let width = size as usize;
        let mut rng = rand::thread_rng();
        let random = rng.gen_range(0.0, 1.0);
        let an = Array::from_shape_fn((height, width), |(i, j)| {
            (((j + i * width) % 3) as f32 - random)
        });
        let bn = Array::from_shape_fn((width, height), |(i, j)| {
            (((j + 7 + i * height) % 3) as f32 + random)
        });
        let a = an.view();
        let b = bn.view();
        for i in 0..ITERS {
            let mut dn = Array::zeros((size, size));
            let start_time = time::precise_time_ns();
            linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut dn);
            let end_time = time::precise_time_ns();
            timeblas[i] = (end_time - start_time) as f64;
        }

        file.write_all(format!("{}\t{}\n", size, average(timeblas),).as_bytes())?;
    }
    Ok(())
}
