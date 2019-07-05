use ndarray::{Array,ArrayView,ArrayViewMut,Ix2,Ix1,Axis,ArrayBase};
use ndarray::LinalgScalar;
use ndarray::linalg;
use std::ops::AddAssign;
use rand::Rng;
use num_traits::identities::Zero;
use num_traits::identities::One;
use ndarray::Data;

pub fn mult<'a, 'b, 'd, A>(
    a: ArrayView<'a, A, Ix2>,
    b: ArrayView<'b, A, Ix2>,
    mut result: ArrayViewMut<'d, A, Ix2>,
)
where
    A: LinalgScalar + AddAssign
{
   for idx_a in 0..a.rows(){
       let arow = a.row(idx_a);
       for idx_b in 0..b.cols(){
           let bcolumn = b.column(idx_b);
            let c = result.get_mut((idx_a,idx_b)).expect("Index out of bounds");
            *c = scalar_mult(arow,bcolumn);
       }
   }
}

fn scalar_mult<'a,'b,A>(a: ArrayView<'a, A, Ix1>,b: ArrayView<'b, A, Ix1>) -> A
where  
    A: LinalgScalar + AddAssign
{
    let mut sum = A::zero();
    for (ea,eb) in a.iter().zip(b.iter()) {
        sum += *ea * *eb;
    }
    sum
}

// i k j
pub fn mult_index_optimized<'a, 'b, 'd, A>(
    a: ArrayView<'a, A, Ix2>,
    b: ArrayView<'b, A, Ix2>,
    mut result: ArrayViewMut<'d, A, Ix2>,
)
where
    A: LinalgScalar + AddAssign
{
    for idx_a in 0..a.rows(){
        let arow = a.row(idx_a);
        for k in 0..a.cols() {
            let r = arow.get(k).expect("index out of bounds");
            let brow = b.row(k);
            for j in 0..b.cols() {
                let c = result.get_mut((idx_a,j)).expect("Index out of bounds");
                let bel = brow.get(j).expect("index out of bounds");
                *c += *r * *bel;
            }  
        }
   }
}

pub fn cut_in_blocks<'a, 'b, 'd, A>(
    a: ArrayView<'a, A, Ix2>,
    b: ArrayView<'b, A, Ix2>,
    mut result: ArrayViewMut<'d, A, Ix2>,
    chunk: usize,
) -> (Vec<Array<A,Ix2>>,Vec<Array<A,Ix2>>,Vec<Array<A,Ix2>>)
where
A: LinalgScalar + AddAssign
{
    
    let (arow,acol) = a.dim();
    let (brow,bcol) = b.dim();
    let (rrow,rcol) = result.dim();

    let avec:Vec<Array<A,Ix2>> = a.axis_chunks_iter(Axis(1), chunk ).map(|row| {
        row.axis_chunks_iter(Axis(0), chunk);
        let aowned = row.into_owned();
        aowned
    }).collect();
     let bvec:Vec<Array<A,Ix2>> = b.axis_chunks_iter(Axis(1), chunk).map(|row| {
        row.axis_chunks_iter(Axis(0), chunk);
        let bowned = row.into_owned();
        bowned
    }).collect();
    let rvec:Vec<Array<A,Ix2>> = result.axis_chunks_iter_mut(Axis(1),chunk).map(|mut row| {
        row.axis_chunks_iter_mut(Axis(0), chunk);
        let rowned = row.into_owned();
        rowned
    }).collect();
    
    (avec,bvec,rvec)
    // let aview =  Array::from_iter(avec.into_iter()).into_shape((((arow/chunk) as f32).floor() as usize,((acol/chunk) as f32).floor() as usize)).unwrap();
    // let bview =  Array::from_iter(bvec.into_iter()).into_shape((((brow/chunk) as f32).floor() as usize,((bcol/chunk) as f32).floor() as usize)).unwrap();
    // let mut rview =  Array::from_iter(rvec.into_iter()).into_shape((((arow/chunk) as f32).floor() as usize,((acol/chunk) as f32).floor() as usize)).unwrap();

    // mult(aview.view(),bview.view(),rview.view_mut());

}

// fn mult_blocks<A>(ablocks:Vec<Array<A,Ix2>>,bblocks:Vec<Array<A,Ix2>>, cblocks:Vec<Array<A,Ix2>> , lena:usize, lenb:usize,lenr:usize)
// where 
// A: LinalgScalar + AddAssign
// {

//     for idx_a in 0..ablocks.len(){
//         let a = ablocks.get(idx_a).expect("out of ranged");
//         for idx_b in 0..bblocks.len(){
//             let mut calc_b = idx_b;
//             if (idx_b/lenb)%2 ==1{
//                 calc_b = lenb -idx_b;
//             }
//             let b = bblocks.get(calc_b).expect("out of ranged");;
//             let mut res = cblocks.get(idx_a * lenb + calc_b).expect("out of ranged");
//             mult_index_optimized(a.view(), b.view(),res.view_mut());
//         }
//    }
// }



// struct MyType<S:LinalgScalar + Data>(ArrayBase<S,Ix2>);

// impl <A:LinalgScalar + Data>Zero for MyType<A> {

// }
#[test]
fn test_mult(){
    let height = 1000;
    let width = 1000;
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let an = Array::from_shape_fn((height, width), |(i, j)| (((j + i * width) % 3) as f32) + random);
    let bn = Array::from_shape_fn((width, height), |(i, j)| (((j + 7 + i * height) % 3) as f32) - random);
    let mut dest = Array::zeros((height, height));
    mult(an.view(),bn.view(),dest.view_mut());
    let mut verif = Array::zeros((height,height));
    linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);
    assert_abs_diff_eq!(dest.as_slice().unwrap() ,verif.as_slice().unwrap(),epsilon = 1e-1f32);

}

#[test]
fn test_mult_indexed_optimized(){
    let height = 1000;
    let width = 1000;
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let an = Array::from_shape_fn((height, width), |(i, j)| (((j + i * width) % 3) as f32) + random);
    let bn = Array::from_shape_fn((width, height), |(i, j)| (((j + 7 + i * height) % 3) as f32) - random);
    let mut dest = Array::zeros((height, height));
    mult_index_optimized(an.view(),bn.view(),dest.view_mut());
    let mut verif = Array::zeros((height,height));
    linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);
    assert_abs_diff_eq!(dest.as_slice().unwrap() ,verif.as_slice().unwrap(),epsilon = 1e-1f32);
}
