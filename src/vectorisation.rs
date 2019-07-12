use faster::*;
use smallvec::SmallVec;
use std::iter;
use ndarray::{ArrayView,ArrayViewMut,Ix2};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use crate::my_ndarray;
#[cfg(test)]
use crate::naive_sequential;
#[cfg(test)]
use ndarray::{linalg,Array};
#[cfg(test)]
use rand::Rng;

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
    let mut i = 0;
    let mut k = 0;
    let mut j = 0;
    while i < stridesa*aheight - (stridesa - awidth) {
            while k < bheight*stridesb - (stridesb-bwidth) {

                multiply_add_local(&mut into[(j)..(j + intowidth)],
                a[i],
                &b[k..(k+bwidth)],
                awidth, 
                aheight, 
                bwidth, 
                bheight, 
                intowidth, 
                intoheight);
                k += stridesb;
                if i % (stridesa) == awidth - 1 {
                    i = i + stridesa - (awidth-1);
                    j = j + stridesinto;
                } else {
                    i += 1;
                }
            }
            k = 0;
    }
}

fn multiply_add_local(    
    mut into: &mut [f32],
    a: f32,
    b: &[f32],
    awidth: usize,
    aheight: usize,
    bwidth: usize,
    bheight: usize,
    intowidth: usize,
    intoheight: usize,
) {
     let aiter = iter::repeat(a).take(bwidth).collect::<SmallVec<[_; 512]>>();
     let temp = (b.simd_iter(f32s(0.)),aiter.simd_iter(f32s(0.)))
     .zip()
     .simd_map(|(x,y)| x*y).scalar_collect();

    let temp = (temp.simd_iter(f32s(0.)),into.simd_iter(f32s(0.)))
            .zip()
            .simd_map(|(x,y)| x + y)
            .scalar_collect();

    temp.simd_iter(f32s(0.)).scalar_fill(&mut into);
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

#[test]
fn test_mult() {
    let height = 4;
    let width = 4;
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let an = Array::from_shape_fn((height, width), |(i, j)| {
        (((j + i * width) % 3) as f32) + random
    });
    let bn = Array::from_shape_fn((width, height), |(i, j)| {
        (((j + 7 + i * height) % 3) as f32) - random
    });
    let mut dn = Array::zeros((height, height));
    let (avec, bvec, rvec) = naive_sequential::cut_in_blocks(an.view(), bn.view(), dn.view_mut(),2,2);
    naive_sequential::mult_blocks(avec, bvec, rvec,|a,b,mut c| mult_faster_from_ndarray(a, b, &mut c));
            
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);

    assert_abs_diff_eq!(
        dn.as_slice().unwrap(),
        verif.as_slice().unwrap(),
        epsilon = 1e-1f32
    );
}