use rayon_adaptive::prelude::*;
use rayon_adaptive::{IndexedPower, Policy};
use ndarray::{ArrayView, ArrayViewMut, Array,Axis};
use ndarray::LinalgScalar;
use ndarray::Ix2;
use ndarray::linalg;
use ndarray::s;
use crate::my_ndarray;


pub struct Matrix<'a,'b,'d,A> {
    pub a: ArrayView<'a,A,Ix2>, 
    pub b: ArrayView<'b,A,Ix2>,
    pub d: ArrayViewMut<'d,A,Ix2>,
}

impl<'a,'b,'d,A>Divisible for Matrix<'a,'b,'d,A> 
where
    A: LinalgScalar + Send + Sync,
    {
    // Can be changed to IndexedPower, need to change divide and divide_mut of my_ndarray
    type Power = IndexedPower;
    
    fn base_length(&self) -> Option<usize> {
        let dim = self.d.shape();
        Some(dim[0]*dim[1])
    }

    fn divide_at(mut self, index: usize) -> (Self,Self) {
        let mut axis = 0;
        let (row,col) = self.d.dim();
        if col > row {
            axis = 1;
        } 

        if axis == 0 {
            let (_ar,ac)= self.a.dim();
            let (_dr,dc) = self.d.dim();
            let (d1,d2) = my_ndarray::divide_mut_at_id_along_axis(self.d, (index - 1), Axis(axis)); 
            let dim_temp = d1.dim().0;  
            let (a1,a2) = my_ndarray::divide_at_id_along_axis(self.a, dim_temp * ac - 1 , Axis(axis));  
            (
                Matrix{
                    a: a1,
                    b: self.b,
                    d: d1,            
                } , 
                Matrix{
                    a: a2,
                    b: self.b,
                    d: d2,
                }
            )
        }
        else {

            let (br,bc)= self.a.dim();
            let (dr,dc) = self.d.dim();
            let ind_calc = ((((index - 1) as f32)/(row as f32)) + 1.0) as usize;
            let (b1,b2) = my_ndarray::divide_at_id_along_axis(self.b, ind_calc*dr/br, Axis(axis));
            let (d1,d2) = my_ndarray::divide_mut_at_id_along_axis(self.d, ind_calc, Axis(axis));
            (
            Matrix{
                    a: self.a,
                    b: b1,
                    d: d1,
                        
                } , 
                Matrix{
                    a: self.a,
                    b: b2,
                    d: d2,
                }
            )
        }
    }
}

#[test]
fn test_mult(){
    let height = 5000;
    let width = 5000;
    let an = Array::from_shape_fn((height, width), |(i, j)| (((j + i * width) % 3) as u32));
    let bn = Array::from_shape_fn((width, height), |(i, j)| (((j + 7 + i * height) % 3) as u32));
    let mut dest = Array::zeros((height, height));
    let mut m = Matrix {
        a: an.view(),
        b: bn.view(),
        d: dest.view_mut(), 
    };
    m.cut().with_policy(Policy::Adaptive(1000,2000))
            .for_each(|e| {
                        let a = e.a;
                        let b = e.b;
                        let mut output = e.d;
                        linalg::general_mat_mul(1u32, &a, &b, 1u32, &mut output);
                    });
    let mut verif = Array::zeros((height,height));
    linalg::general_mat_mul(1u32, &an, &bn, 1u32, &mut verif);
    assert_eq!(dest,verif);

}