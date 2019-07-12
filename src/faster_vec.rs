use crate::my_ndarray;
use faster::*;
use smallvec::SmallVec;
use std::iter;
use std::time::Instant;
use ndarray::{ArrayView,ArrayViewMut,Ix2};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub fn multiply_add(
    into: &mut [f32],
    a: &[f32],
    b: &[f32],
    awidth: usize,
    aheight: usize,
    bwidth: usize,
    bheight: usize,
    intowidth: usize,
    intoheight: usize,
    stridesa: usize,
    stridesb: usize,
    stridesinto: usize,
) {
    assert_eq!(awidth, bheight);
    assert_eq!(aheight, intoheight);
    assert_eq!(bwidth, intowidth);
    let h = intoheight;
    let l = awidth;
    let pads = iter::repeat(f32s(0.))
        .take(stridesb)
        .collect::<SmallVec<[_; 512]>>();
    let columns = b.simd_iter(f32s(0.));
    let columns = columns.stride_into::<SmallVec<[_; 512]>>(stridesb, &pads);
    let mut column_data = iter::repeat(0.0)
        .take(bheight)
        .collect::<SmallVec<[_; 512]>>();
    for (x, mut column) in columns.into_iter().take(bwidth).enumerate() {
        column.scalar_fill(&mut column_data);
        for y in 0..h {
            let row = &a[(y * stridesa)..((y) * stridesa + l)];
            into[((y * stridesinto) + x)] +=
                (row.simd_iter(f32s(0.)), column_data.simd_iter(f32s(0.)))
                    .zip()
                    .simd_reduce(f32s(0.0), |acc, (a, b)| acc + a * b)
                    .sum();
        }
    }
}

pub fn mult_faster_from_ndarray(a: ArrayView<f32,Ix2> ,b: ArrayView<f32,Ix2>,output: &mut ArrayViewMut<f32,Ix2>) {
    let (raw_ptr_a, len_a) = my_ndarray::view_ptr(a);
    let stridesa = a.strides();
    let (raw_ptr_b, len_b) = my_ndarray::view_ptr(b);
    let stridesb = b.strides();
    let raw_ptr_r = output.as_mut_ptr();
    let dimr = output.shape();
    let dima = a.shape();
    let dimb = b.shape();
    let (row, col) = (dimr[0], dimr[1]);
    let strides = output.strides();
    let len_r = (row - 1) * strides[0] as usize + col;
    let slicea = unsafe { from_raw_parts(raw_ptr_a, len_a) };
    let sliceb = unsafe { from_raw_parts(raw_ptr_b, len_b) };
    let mut slicer = unsafe { from_raw_parts_mut(raw_ptr_r, len_r) };
    multiply_add(
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
}
pub fn multiply_add_u32(
    into: &mut [u32],
    a: &[u32],
    b: &[u32],
    awidth: usize,
    aheight: usize,
    bwidth: usize,
    bheight: usize,
    intowidth: usize,
    intoheight: usize,
    stridesa: usize,
    stridesb: usize,
    stridesinto: usize,
) {
    assert_eq!(awidth, bheight);
    assert_eq!(aheight, intoheight);
    assert_eq!(bwidth, intowidth);

    let h = intoheight;
    let l = awidth;

    let pads = iter::repeat(u32s(0))
        .take(stridesb)
        .collect::<SmallVec<[_; 512]>>();
    let columns = b.simd_iter(u32s(0));
    let columns = columns.stride_into::<SmallVec<[_; 512]>>(stridesb, &pads);
    let mut column_data = iter::repeat(0u32)
        .take(bheight)
        .collect::<SmallVec<[_; 512]>>();
    for (x, mut column) in columns.into_iter().take(bwidth).enumerate() {
        column.scalar_fill(&mut column_data);
        for y in 0..h {
            let row = &a[(y * stridesa)..((y) * stridesa + l)];
            into[((y * stridesinto) + x)] +=
                (row.simd_iter(u32s(0)), column_data.simd_iter(u32s(0)))
                    .zip()
                    .simd_reduce(u32s(0), |acc, (a, b)| acc + a * b)
                    .sum();
        }
    }
}

pub fn timed_matmul(size: usize, name: &str, power2: bool) -> u64 {
    let mut size = size;
    if power2 {
        size = size.next_power_of_two();
    }
    let n = size * size;
    let mut a = vec![0f32; n];
    let mut b = vec![0f32; n];

    for i in 0..n {
        a[i] = i as f32;
        b[i] = (i + 7) as f32;
    }
    let mut dest = vec![0f32; n];

    let start = Instant::now();
    multiply_add(
        &mut dest, &a, &b, size, size, size, size, size, size, size, size, size,
    );
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

pub fn timed_matmul_u32(size: usize, name: &str, power2: bool) -> u64 {
    let mut size = size;
    if power2 {
        size = size.next_power_of_two();
    }
    let n = size * size;
    let mut a = vec![0u32; n];
    let mut b = vec![0u32; n];
    for i in 0..n {
        a[i] = i as u32;
        b[i] = (i + 7) as u32;
    }
    let mut dest = vec![0u32; n];
    let a1 = a.as_slice();
    let b1 = b.as_slice();

    let start = Instant::now();
    multiply_add_u32(
        &mut dest, a1, b1, size, size, size, size, size, size, size, size, size,
    );
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