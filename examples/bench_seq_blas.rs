#[macro_use]
extern crate approx;
extern crate rayon;
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
use matrix_mult::vectorisation_packed_simd;
const ITERS: usize = 5;

fn average(numbers: [f64; ITERS as usize]) -> f64 {
    numbers.iter().sum::<f64>() / numbers.len() as f64
}

fn main() -> std::io::Result<()> {
    let input_size: Vec<usize> = (1..17).map(|i| i * 250).collect();
    //do_benchmark_seq("matrix_mult_seq__blas_native", |a,b,mut c|  linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut c),input_size.clone());
    //do_benchmark_seq("matrix_mult_seq_faster_native", |a,b,mut c| faster_vec::mult_faster_from_ndarray(a,b,&mut c),input_size.clone());
    //benchmark_seq_blocks("matrix_mult_seq_faster_600_native",|a,b,mut c| faster_vec::mult_faster_from_ndarray(a,b,&mut c),600,input_size.clone());
    //do_benchmark_seq("matrix_mult_seq_simd_native", |a,b,mut c| vectorisation_packed_simd::mult_faster_from_ndarray(a,b,&mut c),input_size.clone());
    benchmark_seq_blocks("matrix_mult_seq_simd_600_native", |a,b,mut c| vectorisation_packed_simd::mult_faster_from_ndarray(a,b,&mut c),600,input_size.clone());
    Ok(())
}   

fn benchmark_seq_blocks<F>(filename: &str,resolution : F,blocksize:usize, input_size: Vec<usize>) -> std::io::Result<()> 
where
F: Fn(ArrayView<f32, Ix2>, ArrayView<f32, Ix2>, ArrayViewMut<f32,Ix2>) + Copy
{
    let mut file = File::create(filename)?;

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
        for i in 0..ITERS {
            let mut dn = Array::zeros((size, size));
            let start_time = time::precise_time_ns();
            let (avec, bvec, rvec) = naive_sequential::cut_in_blocks(an.view(), bn.view(), dn.view_mut(),blocksize,blocksize);
            naive_sequential::mult_blocks(avec, bvec, rvec,resolution);
            let end_time = time::precise_time_ns();
            let mut verif = Array::zeros((height, height));
            linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);
            assert_abs_diff_eq!(
                dn.as_slice().unwrap(),
                verif.as_slice().unwrap(),
                epsilon = 1e-1f32
            );
            timeblas[i] = (end_time - start_time) as f64;
        }
        file.write_all(format!("{}\t{}\n", size, average(timeblas),).as_bytes())?;
    }
    Ok(())
}

fn do_benchmark_seq<F>(filename: &str,resolution : F, input_size: Vec<usize>) -> std::io::Result<()> 
where
F: Fn(ArrayView<f32, Ix2>, ArrayView<f32, Ix2>, ArrayViewMut<f32,Ix2>) + Copy
{
        let mut file = File::create(filename)?;
      
        for (_j, size) in enumerate(input_size) {
        println!("SIZE : {:?}", size);
        let mut time= [0f64; ITERS as usize];
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
        for i in 0..ITERS {
            let mut dn = Array::zeros((size, size));
            let start_time = time::precise_time_ns();
            resolution(an.view(),bn.view(),dn.view_mut());
            let end_time = time::precise_time_ns();
            
            let mut verif = Array::zeros((height, height));
            linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);
            assert_abs_diff_eq!(
                dn.as_slice().unwrap(),
                verif.as_slice().unwrap(),
                epsilon = 1e-1f32
            );
            time[i] = (end_time - start_time) as f64;
        }
        file.write_all(format!("{}\t{}\n", size, average(time)).as_bytes())?;
    }
    Ok(())
}