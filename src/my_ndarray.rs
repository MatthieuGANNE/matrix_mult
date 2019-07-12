use crate::faster_vec;
use ndarray::linalg;
use ndarray::s;
use ndarray::Array;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::LinalgScalar;
use ndarray::{ArrayView, ArrayViewMut, Axis};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::time::Instant;

const MULT_CHUNK: usize = 1 * 1024;

pub fn timed_matmul_seq_f32(size: usize, name: &str, power2: bool) -> u64 {
    let mut size = size;
    if power2 {
        size = size.next_power_of_two();
    }
    let a = Array::from_shape_fn((size, size), |(i, j)| (i * size + j) as f32);
    let b = Array::from_shape_fn((size, size), |(i, j)| ((i * size) + j + 7) as f32);
    let mut dest = Array::zeros((size, size));

    let start = Instant::now();
    linalg::general_mat_mul(1.0, &a.view(), &b.view(), 1.0, &mut dest.view_mut());
    let dur = Instant::now() - start;
    let nanos = u64::from(dur.subsec_nanos()) + dur.as_secs() * 1_000_000_000u64;
    println!(
        "{}:\t{}x{} matrix: {} s",
        name,
        size,
        size,
        nanos as f32 / 1e9f32
    );
    nanos
}
pub fn timed_matmul_seq_u32(size: usize, name: &str, power2: bool) -> u64 {
    let mut size = size;
    if power2 {
        size = size.next_power_of_two();
    }
    let a = Array::from_shape_fn((size, size), |(i, j)| (i * size + j) as u32);
    let b = Array::from_shape_fn((size, size), |(i, j)| ((i * size) + j + 7) as u32);
    let mut dest = Array::zeros((size, size));

    let start = Instant::now();
    linalg::general_mat_mul(1u32, &a.view(), &b.view(), 1u32, &mut dest.view_mut());
    let dur = Instant::now() - start;
    let nanos = u64::from(dur.subsec_nanos()) + dur.as_secs() * 1_000_000_000u64;
    println!(
        "{}:\t{}x{} matrix: {} s",
        name,
        size,
        size,
        nanos as f32 / 1e9f32
    );
    nanos
}

pub fn timed_matmul_ndarray_f32(size: usize, name: &str, power2: bool) -> u64 {
    let mut size = size;
    if power2 {
        size = size.next_power_of_two();
    }
    let a = Array::from_shape_fn((size, size), |(i, j)| (i * size + j) as f32);
    let b = Array::from_shape_fn((size, size), |(i, j)| ((i * size) + j + 7) as f32);
    let mut dest = Array::zeros((size, size));

    let start = Instant::now();
    mult(a.view(), b.view(), dest.view_mut());
    let dur = Instant::now() - start;
    let nanos = u64::from(dur.subsec_nanos()) + dur.as_secs() * 1_000_000_000u64;
    // let mut verif = Array::zeros((size,size));
    // linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut verif);
    // assert_eq!(dest,verif);
    println!(
        "{}:\t{}x{} matrix: {} s",
        name,
        size,
        size,
        nanos as f32 / 1e9f32
    );
    nanos
}

pub fn timed_matmul_ndarray_u32(size: usize, name: &str, power2: bool) -> u64 {
    let mut size = size;
    if power2 {
        size = size.next_power_of_two();
    }
    let a = Array::from_shape_fn((size, size), |(i, j)| (i * size + j) as u32);
    let b = Array::from_shape_fn((size, size), |(i, j)| ((i * size) + j + 7) as u32);
    let mut dest = Array::zeros((size, size));

    let start = Instant::now();
    mult(a.view(), b.view(), dest.view_mut());
    let dur = Instant::now() - start;
    let nanos = u64::from(dur.subsec_nanos()) + dur.as_secs() * 1_000_000_000u64;
    // let mut verif = Array::zeros((size,size));
    // linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut verif);
    // assert_eq!(dest,verif);
    println!(
        "{}:\t{}x{} matrix: {} s",
        name,
        size,
        size,
        nanos as f32 / 1e9f32
    );
    nanos
}

pub fn timed_matmul_faster_f32(size: usize, name: &str, power2: bool) -> u64 {
    let mut size = size;
    if power2 {
        size = size.next_power_of_two();
    }
    let a = Array::from_shape_fn((size, size), |(i, j)| (i * size + j) as f32);
    let b = Array::from_shape_fn((size, size), |(i, j)| ((i * size) + j + 7) as f32);
    let mut dest = Array::zeros((size, size));

    let start = Instant::now();
    mult_nd_faster(a.view(), b.view(), dest.view_mut());
    let dur = Instant::now() - start;
    let nanos = u64::from(dur.subsec_nanos()) + dur.as_secs() * 1_000_000_000u64;
    // let mut verif = Array::zeros((size,size));
    // linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut verif);
    // assert_eq!(dest,verif);
    println!(
        "{}:\t{}x{} matrix: {} s",
        name,
        size,
        size,
        nanos as f32 / 1e9f32
    );
    nanos
}
pub fn timed_matmul_faster_u32(size: usize, name: &str, power2: bool) -> u64 {
    let mut size = size;
    if power2 {
        size = size.next_power_of_two();
    }
    let a = Array::from_shape_fn((size, size), |(i, j)| (i * size + j) as u32);
    let b = Array::from_shape_fn((size, size), |(i, j)| ((i * size) + j + 7) as u32);
    let mut dest = Array::zeros((size, size));

    let start = Instant::now();
    mult_nd_faster_u32(a.view(), b.view(), dest.view_mut());
    let dur = Instant::now() - start;
    let nanos = u64::from(dur.subsec_nanos()) + dur.as_secs() * 1_000_000_000u64;
    // let mut verif = Array::zeros((size,size));
    // linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut verif);
    // assert_eq!(dest,verif);
    println!(
        "{}:\t{}x{} matrix: {} s",
        name,
        size,
        size,
        nanos as f32 / 1e9f32
    );
    nanos
}
pub fn mult<'a, 'b, 'd, A>(
    a: ArrayView<'a, A, Ix2>,
    b: ArrayView<'b, A, Ix2>,
    mut result: ArrayViewMut<'d, A, Ix2>,
) -> ArrayViewMut<'d, A, Ix2>
where
    A: LinalgScalar + Send + Sync,
{
    let dima = a.shape();
    if dima[0] * dima[1] <= MULT_CHUNK {
        linalg::general_mat_mul(A::one(), &a, &b, A::one(), &mut result);
        return result;
    }
    let (rrow, rcol) = result.dim();
    let (a1, a2, a3, a4) = divide(a);
    let (b1, b2, b3, b4) = divide(b);
    let (d1, d2, d3, d4) = divide_mut(result.slice_mut(s![0..rrow;1,0..rcol;1]));

    let (d, f, g, h) = join4(
        || mult(a1, b1, d1),
        || mult(a1, b2, d2),
        || mult(a3, b1, d3),
        || mult(a3, b2, d4),
    );

    let (_r1, _r2, _r3, _r4) = join4(
        || mult(a2, b3, d),
        || mult(a2, b4, f),
        || mult(a4, b3, g),
        || mult(a4, b4, h),
    );
    result
}

pub fn mult_nd_faster<'a, 'b, 'd>(
    a: ArrayView<'a, f32, Ix2>,
    b: ArrayView<'b, f32, Ix2>,
    mut result: ArrayViewMut<'d, f32, Ix2>,
) -> ArrayViewMut<'d, f32, Ix2> {
    let dima = a.shape();
    let dimb = b.shape();
    if dima[0] == 0 || dima[1] == 0 || dimb[0] == 0 || dimb[1] == 0 {
        return result;
    }

    if dima[0] * dima[1] <= MULT_CHUNK {
        let (raw_ptr_a, len_a) = view_ptr(a);
        let stridesa = a.strides();
        let (raw_ptr_b, len_b) = view_ptr(b);
        let stridesb = b.strides();
        let raw_ptr_r = result.as_mut_ptr();
        let dimr = result.shape();
        let (row, col) = (dimr[0], dimr[1]);
        let strides = result.strides();
        let len_r = (row - 1) * strides[0] as usize + col;

        let slicea = unsafe { from_raw_parts(raw_ptr_a, len_a) };
        let sliceb = unsafe { from_raw_parts(raw_ptr_b, len_b) };
        let mut slicer = unsafe { from_raw_parts_mut(raw_ptr_r, len_r) };
        faster_vec::multiply_add(
            &mut slicer,
            &slicea,
            &sliceb,
            dima[1],
            dima[0],
            dimb[1],
            dimb[0],
            dimr[1],
            dimr[0],
            stridesa[0] as usize,
            stridesb[0] as usize,
            strides[0] as usize,
        );
        return result;
    }

    let (rrow, rcol) = result.dim();
    let (a1, a2, a3, a4) = divide(a);
    let (b1, b2, b3, b4) = divide(b);
    let (d1, d2, d3, d4) = divide_mut(result.slice_mut(s![0..rrow;1,0..rcol;1]));

    let (d, f, g, h) = join4(
        || mult_nd_faster(a1, b1, d1),
        || mult_nd_faster(a1, b2, d2),
        || mult_nd_faster(a3, b1, d3),
        || mult_nd_faster(a3, b2, d4),
    );

    let (_r1, _r2, _r3, _r4) = join4(
        || mult_nd_faster(a2, b3, d),
        || mult_nd_faster(a2, b4, f),
        || mult_nd_faster(a4, b3, g),
        || mult_nd_faster(a4, b4, h),
    );
    result
}
pub fn mult_nd_faster_u32<'a, 'b, 'd>(
    a: ArrayView<'a, u32, Ix2>,
    b: ArrayView<'b, u32, Ix2>,
    mut result: ArrayViewMut<'d, u32, Ix2>,
) -> ArrayViewMut<'d, u32, Ix2> {
    let dim = a.shape();
    let dimb = b.shape();
    if dim[0] * dim[1] <= MULT_CHUNK {
        let (raw_ptr_a, len_a) = view_ptr(a);
        let stridesa = a.strides();
        let (raw_ptr_b, len_b) = view_ptr(b);
        let stridesb = b.strides();
        let raw_ptr_r = result.as_mut_ptr();
        let dimr = result.shape();
        let (row, col) = (dimr[0], dimr[1]);
        let strides = result.strides();
        let len_r = (row - 1) * strides[0] as usize + col;

        let slicea = unsafe { from_raw_parts(raw_ptr_a, len_a) };
        let sliceb = unsafe { from_raw_parts(raw_ptr_b, len_b) };
        let mut slicer = unsafe { from_raw_parts_mut(raw_ptr_r, len_r) };
        faster_vec::multiply_add_u32(
            &mut slicer,
            &slicea,
            &sliceb,
            dim[1],
            dim[0],
            dimb[1],
            dimb[0],
            dimr[1],
            dimr[0],
            stridesa[0] as usize,
            stridesb[0] as usize,
            strides[0] as usize,
        );
        return result;
    }
    let (rrow, rcol) = result.dim();
    let (a1, a2, a3, a4) = divide(a);
    let (b1, b2, b3, b4) = divide(b);
    let (d1, d2, d3, d4) = divide_mut(result.slice_mut(s![0..rrow;1,0..rcol;1]));

    let (d, f, g, h) = join4(
        || mult_nd_faster_u32(a1, b1, d1),
        || mult_nd_faster_u32(a1, b2, d2),
        || mult_nd_faster_u32(a3, b1, d3),
        || mult_nd_faster_u32(a3, b2, d4),
    );

    let (_r1, _r2, _r3, _r4) = join4(
        || mult_nd_faster_u32(a2, b3, d),
        || mult_nd_faster_u32(a2, b4, f),
        || mult_nd_faster_u32(a4, b3, g),
        || mult_nd_faster_u32(a4, b4, h),
    );
    result
}

pub fn divide_mut<'a: 'b, 'b, A>(
    a: ArrayViewMut<'a, A, Ix2>,
) -> (
    ArrayViewMut<'a, A, Ix2>,
    ArrayViewMut<'a, A, Ix2>,
    ArrayViewMut<'a, A, Ix2>,
    ArrayViewMut<'a, A, Ix2>,
) {
    let dim = a.shape();
    let (arow, acol) = (dim[0], dim[1]);
    let l_row = arow / 2;
    let l_col = acol / 2;
    let (a1, a2) = a.split_at(Axis(0), l_row);
    let (a11, a12) = a1.split_at(Axis(1), l_col);
    let (a21, a22) = a2.split_at(Axis(1), l_col);
    (a11, a12, a21, a22)
}

pub fn divide<'a, A, D>(
    a: ArrayView<'a, A, D>,
) -> (
    ArrayView<'a, A, D>,
    ArrayView<'a, A, D>,
    ArrayView<'a, A, D>,
    ArrayView<'a, A, D>,
)
where
    D: Dimension,
{
    let dim = a.shape();

    let (arow, acol) = (dim[0], dim[1]);
    let l_row = arow / 2;
    let l_col = acol / 2;

    let (a1, a2) = a.split_at(Axis(0), l_row);
    let (a11, a12) = a1.split_at(Axis(1), l_col);
    let (a21, a22) = a2.split_at(Axis(1), l_col);
    (a11, a12, a21, a22)
}

pub fn divide_at_id_along_axis<'a, A>(
    a: ArrayView<'a, A, Ix2>,
    index: usize,
    axis: Axis,
) -> (ArrayView<'a, A, Ix2>, ArrayView<'a, A, Ix2>) {
    let dim = a.shape();
    let (_arow, acol) = (dim[0], dim[1]);
    if axis.index() == 0 {
        let l_row = index / acol + 1;
        let (a1, a2) = a.split_at(axis, l_row);
        (a1, a2)
    } else {
        let l_col = index % acol;
        let (a1, a2) = a.split_at(axis, l_col);
        (a1, a2)
    }
}

pub fn divide_mut_at_id_along_axis<'a, A>(
    a: ArrayViewMut<'a, A, Ix2>,
    index: usize,
    axis: Axis,
) -> (ArrayViewMut<'a, A, Ix2>, ArrayViewMut<'a, A, Ix2>) {
    let dim = a.shape();

    let (_arow, acol) = (dim[0], dim[1]);
    if axis.index() == 0 {
        let l_row = index / acol + 1;
        let (a1, a2) = a.split_at(axis, l_row);
        (a1, a2)
    } else {
        let l_col = index % acol;
        let (a1, a2) = a.split_at(axis, l_col);
        (a1, a2)
    }
}

fn join4<F1, F2, F3, F4, R1, R2, R3, R4>(f1: F1, f2: F2, f3: F3, f4: F4) -> (R1, R2, R3, R4)
where
    F1: FnOnce() -> R1 + Send,
    R1: Send,
    F2: FnOnce() -> R2 + Send,
    R2: Send,
    F3: FnOnce() -> R3 + Send,
    R3: Send,
    F4: FnOnce() -> R4 + Send,
    R4: Send,
{
    let ((r1, r2), (r3, r4)) = rayon::join(|| rayon::join(f1, f2), || rayon::join(f3, f4));
    (r1, r2, r3, r4)
}

pub fn view_ptr<A>(view: ArrayView<A, Ix2>) -> (*const A, usize)
where
    A: LinalgScalar,
{
    let dim = view.shape();
    let (row, _col) = (dim[0], dim[1]);
    let raw_ptr = view.as_ptr();
    let strides = view.strides();
    let len = row * strides[0] as usize;

    (raw_ptr, len)
}
#[test]
fn test_mult() {
    let a = Array::from_shape_fn((25, 25), |(i, j)| i + j);
    let b = Array::from_shape_fn((25, 25), |(i, j)| i + j);
    let (a_r, _a_c) = a.dim();
    let (_b_r, b_c) = b.dim();
    let mut result = Array::zeros((a_r, b_c));

    mult(a.view(), b.view(), result.view_mut());
    let mut verif = Array::zeros((25, 25));
    linalg::general_mat_mul(1, &a, &b, 1, &mut verif);
    assert_eq!(verif, result);
}
