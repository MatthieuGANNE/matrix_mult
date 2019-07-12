#[macro_use]
extern crate criterion;
extern crate rayon;
extern crate rayon_adaptive;

use criterion::{Criterion, ParameterizedBenchmark};
use matrix_mult::matrix;
use matrix_mult::my_ndarray;
use ndarray::s;
use ndarray::{Array, ArrayView, ArrayViewMut};
use rand::Rng;
use rayon_adaptive::prelude::*;
use rayon_adaptive::Policy;

fn all_adaptive(c: &mut Criterion) {
    let sizes = vec![5, 10, 25, 50, 100, 500, 1000, 1500, 2000, 2500, 3000];
    c.bench(
        "mult",
        ParameterizedBenchmark::new(
            "Rayon 1",
            |b, input_size| {
                b.iter_with_setup(
                    || {
                        let height = *input_size as usize;
                        let width = *input_size as usize;
                        let an = Array::from_shape_fn((height, width), |(i, j)| {
                            (((j + i * width) % 3) as f32)
                        });
                        let bn = Array::from_shape_fn((width, height), |(i, j)| {
                            (((j + 7 + i * height) % 3) as f32)
                        });
                        let dest = Array::zeros((height, height));
                        (an, bn, dest)
                    },
                    |m| {
                        let (an, bn, mut dest) = m;
                        let mut vec = Vec::new();
                        vec.push(([(an.view(), bn.view())].to_vec(), dest.view_mut()));
                        let mat = matrix::Matrix { matrix: vec };
                        mat.cut().with_policy(Policy::Rayon(1)).for_each(|e| {
                            for (vect, mut output) in e.matrix {
                                let (rrow, rcol) = output.dim();
                                for (a, b) in vect {
                                    let o = output.slice_mut(s![0..rrow;1,0..rcol;1]);
                                    my_ndarray::mult(a, b, o);
                                }
                            }
                        });
                    },
                )
            },
            sizes,
        )
        .with_function("Rayon 1000", |b, input_size| {
            b.iter_with_setup(
                || {
                    let height = *input_size as usize;
                    let width = *input_size as usize;
                    let an = Array::from_shape_fn((height, width), |(i, j)| {
                        (((j + i * width) % 3) as f32)
                    });
                    let bn = Array::from_shape_fn((width, height), |(i, j)| {
                        (((j + 7 + i * height) % 3) as f32)
                    });
                    let dest = Array::zeros((height, height));
                    (an, bn, dest)
                },
                |m| {
                    let (an, bn, mut dest) = m;
                    let mut vec = Vec::new();
                    vec.push(([(an.view(), bn.view())].to_vec(), dest.view_mut()));
                    let mat = matrix::Matrix { matrix: vec };
                    mat.cut().with_policy(Policy::Rayon(1000)).for_each(|e| {
                        for (vect, mut output) in e.matrix {
                            let (rrow, rcol) = output.dim();
                            for (a, b) in vect {
                                let o = output.slice_mut(s![0..rrow;1,0..rcol;1]);
                                my_ndarray::mult(a, b, o);
                            }
                        }
                    });
                },
            )
        })
        .with_function("Sequential", |b, input_size| {
            b.iter_with_setup(
                || {
                    let height = *input_size as usize;
                    let width = *input_size as usize;
                    let an = Array::from_shape_fn((height, width), |(i, j)| {
                        (((j + i * width) % 3) as f32)
                    });
                    let bn = Array::from_shape_fn((width, height), |(i, j)| {
                        (((j + 7 + i * height) % 3) as f32)
                    });
                    let dest = Array::zeros((height, height));
                    (an, bn, dest)
                },
                |m| {
                    let (an, bn, mut dest) = m;
                    let mut vec = Vec::new();
                    vec.push(([(an.view(), bn.view())].to_vec(), dest.view_mut()));
                    let mat = matrix::Matrix { matrix: vec };
                    mat.cut().with_policy(Policy::Sequential).for_each(|e| {
                        for (vect, mut output) in e.matrix {
                            let (rrow, rcol) = output.dim();
                            for (a, b) in vect {
                                let o = output.slice_mut(s![0..rrow;1,0..rcol;1]);
                                my_ndarray::mult(a, b, o);
                            }
                        }
                    });
                },
            )
        })
        .with_function("Join", |b, input_size| {
            b.iter_with_setup(
                || {
                    let height = *input_size as usize;
                    let width = *input_size as usize;
                    let an = Array::from_shape_fn((height, width), |(i, j)| {
                        (((j + i * width) % 3) as f32)
                    });
                    let bn = Array::from_shape_fn((width, height), |(i, j)| {
                        (((j + 7 + i * height) % 3) as f32)
                    });
                    let dest = Array::zeros((height, height));
                    (an, bn, dest)
                },
                |m| {
                    let (an, bn, mut dest) = m;
                    let mut vec = Vec::new();
                    vec.push(([(an.view(), bn.view())].to_vec(), dest.view_mut()));
                    let mat = matrix::Matrix { matrix: vec };
                    mat.cut()
                        .with_policy(Policy::Join((input_size / 64 + 1) as usize))
                        .for_each(|e| {
                            for (vect, mut output) in e.matrix {
                                let (rrow, rcol) = output.dim();
                                for (a, b) in vect {
                                    let o = output.slice_mut(s![0..rrow;1,0..rcol;1]);
                                    my_ndarray::mult(a, b, o);
                                }
                            }
                        });
                },
            )
        }),
    );
}

criterion_group!(benches, all_adaptive);
criterion_main!(benches);
