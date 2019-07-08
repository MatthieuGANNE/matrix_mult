use crate::my_ndarray;
use crate::split::split;
use ndarray::LinalgScalar;
#[cfg(test)]
use ndarray::{linalg, Array};
use ndarray::{ArrayView, ArrayViewMut, Axis, Ix1, Ix2};
#[cfg(test)]
use rand::Rng;
use rayon_adaptive::prelude::*;
use rayon_adaptive::Policy;
use std::ops::AddAssign;

pub fn mult<'a, 'b, 'd, A>(
    a: ArrayView<'a, A, Ix2>,
    b: ArrayView<'b, A, Ix2>,
    mut result: ArrayViewMut<'d, A, Ix2>,
) where
    A: LinalgScalar + AddAssign,
{
    for idx_a in 0..a.rows() {
        let arow = a.row(idx_a);
        for idx_b in 0..b.cols() {
            let bcolumn = b.column(idx_b);
            let c = result.get_mut((idx_a, idx_b)).expect("Index out of bounds");
            *c = scalar_mult(arow, bcolumn);
        }
    }
}

fn scalar_mult<'a, 'b, A>(a: ArrayView<'a, A, Ix1>, b: ArrayView<'b, A, Ix1>) -> A
where
    A: LinalgScalar + AddAssign,
{
    let mut sum = A::zero();
    for (ea, eb) in a.iter().zip(b.iter()) {
        sum += *ea * *eb;
    }
    sum
}

// i k j
pub fn mult_index_optimized<'a, 'b, 'd, A>(
    a: ArrayView<'a, A, Ix2>,
    b: ArrayView<'b, A, Ix2>,
    mut result: ArrayViewMut<'d, A, Ix2>,
) where
    A: LinalgScalar + AddAssign,
{
    for idx_a in 0..a.rows() {
        let arow = a.row(idx_a);
        for k in 0..a.cols() {
            let r = arow.get(k).expect("index out of bounds");
            let brow = b.row(k);
            for j in 0..b.cols() {
                let c = result.get_mut((idx_a, j)).expect("Index out of bounds");
                let bel = brow.get(j).expect("index out of bounds");
                *c += *r * *bel;
            }
        }
    }
}

pub fn cut_in_blocks<'a, 'b, 'd, A>(
    a: ArrayView<'a, A, Ix2>,
    b: ArrayView<'b, A, Ix2>,
    result: ArrayViewMut<'d, A, Ix2>,
    chunkw: usize,
    chunkh: usize,
) -> (
    Vec<Vec<ArrayView<'a, A, Ix2>>>,
    Vec<Vec<ArrayView<'b, A, Ix2>>>,
    Vec<Vec<ArrayViewMut<'d, A, Ix2>>>,
)
where
    A: LinalgScalar + AddAssign + Sync + Send,
{
    let (_arow, acol) = a.dim();
    let (_brow, bcol) = b.dim();
    let (_rrow, rcol) = result.dim();
    let avec_blocks: Vec<Vec<ArrayView<'a, A, Ix2>>> = split(
        a,
        |a| {
            let (ar, _ac) = a.dim();
            my_ndarray::divide_at_id_along_axis(a, ar / 2 * acol - 1, Axis(0))
        },
        |a| {
            let (ar, _ac) = a.dim();
            ar
        },
    )
    .cut()
    .with_policy(Policy::Join(chunkh))
    .map(|subblock| {
        split(
            subblock.data,
            |s| {
                let (_ar, ac) = s.dim();
                my_ndarray::divide_at_id_along_axis(s, ac / 2, Axis(1))
            },
            |a| {
                let (_ar, ac) = a.dim();
                ac
            },
        )
        .with_policy(Policy::Join(chunkw))
        .collect()
    })
    .collect();

    let bvec_blocks: Vec<Vec<ArrayView<'b, A, Ix2>>> = split(
        b,
        |b| {
            let (br, _bc) = b.dim();
            my_ndarray::divide_at_id_along_axis(b, br / 2 * bcol - 1, Axis(0))
        },
        |b| {
            let (br, _bc) = b.dim();
            br
        },
    )
    .cut()
    .with_policy(Policy::Join(chunkw))
    .map(|subblock| {
        split(
            subblock.data,
            |s| {
                let (_br, bc) = s.dim();
                my_ndarray::divide_at_id_along_axis(s, bc / 2, Axis(1))
            },
            |b| {
                let (_br, bc) = b.dim();
                bc
            },
        )
        .with_policy(Policy::Join(chunkh))
        .collect()
    })
    .collect();
    let rvec_blocks: Vec<Vec<ArrayViewMut<'d, A, Ix2>>> = split(
        result,
        |r| {
            let (rr, rc) = r.dim();
            my_ndarray::divide_mut_at_id_along_axis(r, rr / 2 * rcol - 1, Axis(0))
        },
        |r| {
            let (rr, _rc) = r.dim();
            rr
        },
    )
    .cut()
    .with_policy(Policy::Join(chunkh))
    .map(|subblock| {
        split(
            subblock.data,
            |s| {
                let (rr, rc) = s.dim();
                my_ndarray::divide_mut_at_id_along_axis(s, rc / 2, Axis(1))
            },
            |r| {
                let (_rr, rc) = r.dim();
                rc
            },
        )
        .with_policy(Policy::Join(chunkh))
        .collect()
    })
    .collect();
    (avec_blocks, bvec_blocks, rvec_blocks)
}

pub fn mult_blocks<A>(
    ablocks: Vec<Vec<ArrayView<A, Ix2>>>,
    bblocks: Vec<Vec<ArrayView<A, Ix2>>>,
    mut cblocks: Vec<Vec<ArrayViewMut<A, Ix2>>>,
) where
    A: LinalgScalar + AddAssign,
{
    for line_a in 0..ablocks.len() {
        let aline = ablocks.get(line_a).expect("out of range in Matrix A");
        for (idx_a, line_b) in (0..aline.len()).zip(0..bblocks.len()) {
            let a = aline.get(idx_a).expect("out of range in Matrix A");
            let bline = bblocks.get(line_b).expect("out of range in Matrix B");
            let nb_col_b = bline.len();
            for idx_b in 0..bline.len() {
                let mut calc_b = idx_b;
                if (line_b / nb_col_b) % 2 == 1 {
                    calc_b = nb_col_b - idx_b - 1;
                }
                let b = bline.get(calc_b).expect("out of range in Matrix B");
                let resline = cblocks
                    .get_mut(line_a)
                    .expect("out of range in Matrix Result");
                let res = resline
                    .get_mut(calc_b)
                    .expect("out of range in Matrix Result");
                mult_index_optimized(a.view(), b.view(), res.view_mut());
            }
        }
    }
}

#[test]
fn test_mult() {
    let height = 1000;
    let width = 1000;
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let an = Array::from_shape_fn((height, width), |(i, j)| {
        (((j + i * width) % 3) as f32) + random
    });
    let bn = Array::from_shape_fn((width, height), |(i, j)| {
        (((j + 7 + i * height) % 3) as f32) - random
    });
    let mut dest = Array::zeros((height, height));
    mult(an.view(), bn.view(), dest.view_mut());
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);
    assert_abs_diff_eq!(
        dest.as_slice().unwrap(),
        verif.as_slice().unwrap(),
        epsilon = 1e-1f32
    );
}

#[test]
fn test_mult_indexed_optimized() {
    let height = 1000;
    let width = 1000;
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let an = Array::from_shape_fn((height, width), |(i, j)| {
        (((j + i * width) % 3) as f32) + random
    });
    let bn = Array::from_shape_fn((width, height), |(i, j)| {
        (((j + 7 + i * height) % 3) as f32) - random
    });
    let mut dest = Array::zeros((height, height));
    mult_index_optimized(an.view(), bn.view(), dest.view_mut());
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);
    assert_abs_diff_eq!(
        dest.as_slice().unwrap(),
        verif.as_slice().unwrap(),
        epsilon = 1e-1f32
    );
}

#[test]
fn test_mult_blocks() {
    let height = 1000;
    let width = 1000;
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let an = Array::from_shape_fn((height, width), |(i, j)| {
        (((j + i * width) % 3) as f32) + random
    });
    let bn = Array::from_shape_fn((width, height), |(i, j)| {
        (((j + 7 + i * height) % 3) as f32) - random
    });
    let mut dest = Array::zeros((height, height));
    let (avec, bvec, rvec) = cut_in_blocks(an.view(), bn.view(), dest.view_mut(), 10, 10);
    mult_blocks(avec, bvec, rvec);
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);
    assert_abs_diff_eq!(
        dest.as_slice().unwrap(),
        verif.as_slice().unwrap(),
        epsilon = 1e-1f32
    );
}
