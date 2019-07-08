extern crate rayon;
use itertools::enumerate;
use matrix_mult::my_ndarray;
use matrix_mult::rayon_mult;
use std::fs::File;
use std::io::Write;
const ITERS: usize = 50;

fn average(numbers: [f64; ITERS as usize]) -> f64 {
    numbers.iter().sum::<f64>() / numbers.len() as f64
}

fn median(numbers: [f64; ITERS as usize]) -> f64 {
    let mid = numbers.len() / 2;
    numbers[mid]
}
fn q1(numbers: [f64; ITERS as usize]) -> f64 {
    let mid = numbers.len() / 4;
    numbers[mid]
}
fn q3(numbers: [f64; ITERS as usize]) -> f64 {
    let mid = numbers.len() * 3 / 4;
    numbers[mid]
}

fn max(numbers: [f64; ITERS as usize]) -> f64 {
    numbers[numbers.len() - 1]
}

fn min(numbers: [f64; ITERS as usize]) -> f64 {
    numbers[0]
}

fn main() -> std::io::Result<()> {
    let mut file = File::create("algo1.data")?;
    let mut filescheduling = File::create("algo2.data")?;
    let input_size = vec![50, 100, 200, 500, 1000, 2000];

    for (j, size) in enumerate(input_size) {
        let mut vecalgo1 = [0f64; ITERS as usize];
        let mut vecalgo2 = [0f64; ITERS as usize];

        for i in 0..ITERS {
            let seq = rayon_mult::timed_matmul(size, rayon_mult::seq_matmulz, "seq z-order");
            let par = rayon_mult::timed_matmul(size, rayon_mult::matmulz, "par z-order");
            let par_nd = my_ndarray::timed_matmul_ndarray_f32(size, "par ndarray", true);
            let mut speedup = seq as f64 / par as f64;
            vecalgo1[i] = speedup;
            speedup = seq as f64 / par_nd as f64;
            vecalgo2[i] = speedup;
        }
        vecalgo1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        file.write_all(
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                j,
                min(vecalgo1),
                q1(vecalgo1),
                median(vecalgo1),
                q3(vecalgo1),
                max(vecalgo1),
                0.3,
                size,
                average(vecalgo1)
            )
            .as_bytes(),
        )?;
        vecalgo2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        filescheduling.write_all(
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                j,
                min(vecalgo2),
                q1(vecalgo2),
                median(vecalgo2),
                q3(vecalgo2),
                max(vecalgo2),
                0.3,
                size,
                average(vecalgo2)
            )
            .as_bytes(),
        )?;
    }
    Ok(())
}
