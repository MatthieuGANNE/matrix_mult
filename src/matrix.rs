use rayon_adaptive::prelude::*;
use rayon_adaptive::{BasicPower, Policy};
use ndarray::{ArrayView, ArrayViewMut, Array};
use ndarray::LinalgScalar;
use ndarray::Ix2;
use ndarray::linalg;
use ndarray::s;
use rand::Rng;
use crate::my_ndarray;


pub struct Matrix<'a,'b,'d,A> {
    pub matrix :Vec<(Vec<( ArrayView<'a,A,Ix2>, ArrayView<'b,A,Ix2>)>,ArrayViewMut<'d,A,Ix2>)>,
}

impl<'a,'b,'d,A>Divisible for Matrix<'a,'b,'d,A> 
where
    A: LinalgScalar + Send + Sync,
    {
    // Can be changed to IndexedPower, need to change divide and divide_mut of my_ndarray
    type Power = BasicPower;
    
    fn base_length(&self) -> Option<usize> {
        Some(self.matrix.iter().map(|m| m.1.shape().iter().product::<usize>()).sum::<usize>())
    }

    fn divide_at(mut self, index: usize) -> (Self,Self) {
        if self.matrix.is_empty() {
            let other: Vec<(Vec<(ArrayView<A,Ix2>,ArrayView<A,Ix2>)>,ArrayViewMut<A,Ix2>)> = Vec::new();
           (
               self,
               Matrix {
                   matrix: other
               }
           )
        }
        else {
            let mut other : Vec<(Vec<(ArrayView<A,Ix2>,ArrayView<A,Ix2>)>,ArrayViewMut<A,Ix2>)> = Vec::new();
            if self.matrix.len() == 1 {
                let mut copy_self : Vec<(Vec<(ArrayView<A,Ix2>,ArrayView<A,Ix2>)>,ArrayViewMut<A,Ix2>)> = Vec::new();
                    let sub_matrix = self.matrix.pop().unwrap();
                    let (v1,v2,v3,v4) = divide(sub_matrix);
                    copy_self.push(v1);
                    copy_self.push(v2);
                    other.push(v3);
                    other.push(v4);             
                (
                    Matrix {
                        matrix: copy_self,
                    }
                    ,
                    Matrix {
                        matrix: other,
                    }
                )
            } else {
                other.push(self.matrix.pop().unwrap());
                (
                    self,
                    Matrix {
                        matrix: other,
                    }
                )   
            }
            
        }
    }
}

fn divide<'a,'b,'c,A>(mut sub_matrix:(Vec<(ArrayView<'a,A,Ix2>,ArrayView<'b,A,Ix2>)>,ArrayViewMut<'c,A,Ix2>) ) ->
((Vec<(ArrayView<'a,A,Ix2>,ArrayView<'b,A,Ix2>)>,ArrayViewMut<'c,A,Ix2>),
(Vec<(ArrayView<'a,A,Ix2>,ArrayView<'b,A,Ix2>)>,ArrayViewMut<'c,A,Ix2>),
(Vec<(ArrayView<'a,A,Ix2>,ArrayView<'b,A,Ix2>)>,ArrayViewMut<'c,A,Ix2>),
(Vec<(ArrayView<'a,A,Ix2>,ArrayView<'b,A,Ix2>)>,ArrayViewMut<'c,A,Ix2>),
)
where 
A: LinalgScalar + Send + Sync,
{
    let mut r1: Vec<(ArrayView<A,Ix2>,ArrayView<A,Ix2>)> = Vec::new();
    let mut r2: Vec<(ArrayView<A,Ix2>,ArrayView<A,Ix2>)> = Vec::new();
    let mut r3: Vec<(ArrayView<A,Ix2>,ArrayView<A,Ix2>)> = Vec::new();
    let mut r4: Vec<(ArrayView<A,Ix2>,ArrayView<A,Ix2>)> = Vec::new();
    let (d1,d2,d3,d4) = my_ndarray::divide_mut(sub_matrix.1);
    while(!sub_matrix.0.is_empty()) {
        let (a,b)= sub_matrix.0.pop().unwrap();
    
        let (a1,a2,a3,a4) = my_ndarray::divide(a);
        let (b1,b2,b3,b4) = my_ndarray::divide(b);
        r1.push((a1,b1));
        r1.push((a2,b3));
        r2.push((a1,b2));
        r2.push((a2,b4)); 
        r3.push((a3,b1));
        r3.push((a4,b3));
        r4.push((a3,b2));
        r4.push((a4,b4));
    }
    ((r1,d1),(r2,d2),(r3,d3),(r4,d4))
}



#[test]
fn test_mult(){
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let height = 1000;
    let width = 1000;
    let an = Array::from_shape_fn((height, width), |(i, j)| (((j + i * width) % 3) as f32) + random);
    let bn = Array::from_shape_fn((width, height), |(i, j)| (((j + 7 + i * height) % 3) as f32) - random);
    let mut dest = Array::zeros((height, height));
    let mut m = Matrix {
        matrix: Vec::new()
    };
    m.matrix.push(([(an.view(),bn.view())].to_vec(),dest.view_mut()));
    m.cut().with_policy(Policy::Rayon(1000))
            .for_each(|e| {
                for (vect,mut output) in e.matrix {
                    for (a,b) in  vect {
                        linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut output);
                    }
                }
            });
    let mut verif = Array::zeros((height,height));
    linalg::general_mat_mul(1.0, &an, &bn, 1.0, &mut verif);
    assert_abs_diff_eq!(dest.as_slice().unwrap() ,verif.as_slice().unwrap(),epsilon = 1e-1f32);

}