use crate::my_ndarray;
use ndarray::Ix2;
use ndarray::LinalgScalar;
#[cfg(test)]
use ndarray::{linalg, Array};
use ndarray::{ArrayView, ArrayViewMut, Axis,ShapeBuilder};
#[cfg(test)]
use rand::Rng;
use rayon_adaptive::prelude::*;
use rayon_adaptive::IndexedPower;
use std::iter::{once, Once};

#[cfg(test)]
use rayon_adaptive::Policy;

pub struct Matrix<'a, 'b, 'd, A> {
    pub a: ArrayView<'a, A, Ix2>,
    pub b: ArrayView<'b, A, Ix2>,
    pub d: ArrayViewMut<'d, A, Ix2>,
    pub asize : (usize,usize),
    pub bsize : (usize,usize),
    pub dsize : (usize,usize),
}

impl<'a, 'b, 'd, A> Divisible for Matrix<'a, 'b, 'd, A>
where
    A: LinalgScalar + Send + Sync,
{
    // Can be changed to IndexedPower, need to change divide and divide_mut of my_ndarray
    type Power = IndexedPower;

    fn base_length(&self) -> Option<usize> {
        let dim = self.d.shape();
        let dima = self.a.shape();
        if dima[1]<=1 {
            Some(1)
        }
        else {
            Some(dim[0] * dim[1])
        }
    }

    fn divide(self) -> (Self,Self) {
        
        let mut axis = 0;
        let (row, col) = self.d.dim();
        if col > row {
            axis = 1;
        }

        if axis == 0 {
            let mut r2 = row;
             if !row.is_power_of_two() {
                r2 = row.next_power_of_two();
            }
            let rowbef =  r2 / 4;
            let rownext = r2 / 2;
            let diff1 = row / 2 - rowbef;
            let diff2 = rownext - row / 2;
            if diff1 < diff2 {
                self.divide_at(rowbef * col)
            }
            else {
                self.divide_at(rownext * col)
            }
        } else {
            let mut c2 = col;
            if !col.is_power_of_two(){
                c2 = col.next_power_of_two();
            }
            let colbef = c2 / 4;
            let colnext = c2 / 2;
            let diff1 = col / 2 - colbef;
            let diff2 = colnext - col / 2;
            if diff1 < diff2 {
                self.divide_at(colbef * row)
            }
            else {

                self.divide_at(colnext * row)
            }
        }
    }
    #[allow(unused_mut)]
    fn divide_at(mut self, index: usize) -> (Self, Self) {
        let mut axis = 0;
        let (row, col) = self.d.dim();
        if col > row {
            axis = 1;
        }
        if axis == 0 {
            let (ar, ac) = self.a.dim();
            let (d1, d2) = my_ndarray::divide_mut_at_id_along_axis(self.d, index - 1, Axis(axis));
            let dim_temp = d1.dim().0;
            let (a1, a2) =
                my_ndarray::divide_at_id_along_axis(self.a, dim_temp * ac, Axis(axis));
            let (ra1,ca1) = a1.dim();
            let (rd1,cd1) = d1.dim();
            let (ra2,ca2) = a2.dim();
            let (rd2,cd2) = d2.dim();
            let (rb,cb) = self.b.dim();
            if ra1 != 0 && ca1 !=0 && ra2 != 0 && ca2 !=0 && rd1 != 0 && cd1 !=0 && rd2 != 0 && cd2 !=0 &&  rb != 0 && cb !=0 {
                assert_eq!(rd1,ra1);
                assert_eq!(rd2,ra2);
                assert_eq!(ca1,rb);
                assert_eq!(ca2,rb);
                assert_eq!(cd1,cb);
                assert_eq!(cd2,cb);
            }
            (
                Matrix {
                    a: a1,
                    b: self.b,
                    d: d1,
                    asize: self.asize,
                    bsize: self.bsize,
                    dsize: self.dsize,
                },
                Matrix {
                    a: a2,
                    b: self.b,
                    d: d2,
                    asize: self.asize,
                    bsize: self.bsize,
                    dsize: self.dsize,
                },
            )
        } else {
            let ind_calc = ((((index - 1) as f32) / (row as f32)) + 1.0) as usize;
            let (d1, d2) = my_ndarray::divide_mut_at_id_along_axis(self.d, ind_calc, Axis(axis));
            let (b1, b2) =
                my_ndarray::divide_at_id_along_axis(self.b, ind_calc, Axis(axis));
            let (rb1,cb1) = b1.dim();
            let (rd1,cd1) = d1.dim();
            let (rb2,cb2) = b2.dim();
            let (rd2,cd2) = d2.dim();
            let (ra,ca) = self.a.dim();
            if rb1 != 0 && cb1 !=0 && rb2 != 0 && cb2 !=0 && rd1 != 0 && cd1 !=0 && rd2 != 0 && cd2 !=0 &&  ra != 0 && ca !=0 {
                assert_eq!(rd1,ra);
                assert_eq!(rd2,ra);
                assert_eq!(ca,rb1);
                assert_eq!(ca,rb2);
                assert_eq!(cd1,cb1);
                assert_eq!(cd2,cb2);
            }
            (
                Matrix {
                    a: self.a,
                    b: b1,
                    d: d1,
                    asize: self.asize,
                    bsize: self.bsize,
                    dsize: self.dsize,
                },
                Matrix {
                    a: self.a,
                    b: b2,
                    d: d2,
                    asize: self.asize,
                    bsize: self.bsize,
                    dsize: self.dsize,
                },
            )
        }
    }
}


impl<'a, 'b, 'd, A> ParallelIterator for Matrix<'a, 'b, 'd, A>
where
    A: LinalgScalar + Send + Sync,
{
    type Item = Self;

    type SequentialIterator = Once<Self>;

    fn to_sequential(self) -> Self::SequentialIterator {
        once(self)
    }

    fn extract_iter(&mut self, size: usize) -> Self::SequentialIterator {
        let (x2,z2) = self.dsize;
        let (x,y) = self.asize;
        let (y1,z1) = self.bsize;
        let (ar,ac) = self.a.dim();
        let (br,bc) = self.b.dim();

        let stridesd = self.d.strides();
        let (f,g) = (stridesd[0],stridesd[1]);
        
        let mut idx = (size* y) / (ar * bc) as usize;
        idx = idx.next_power_of_two() / 2;
        if idx == 0 {
            idx = 1;
        }

        if idx > ac {
            idx = ac;
        }

        // Cut along the column
        let (a1, a2) =
        my_ndarray::divide_at_id_along_axis(self.a, idx, Axis(1));
        // Cut alog the row
        let (b1, b2) =
                my_ndarray::divide_at_id_along_axis(self.b, (idx) * bc - 1, Axis(0));
        let (rd,cd) = self.d.dim();
        let raw_mut= &mut self.d[[0,0]];
        let cpy_d:ArrayViewMut<A,Ix2> = unsafe{ArrayViewMut::from_shape_ptr((rd,cd).strides((f as usize ,g as usize)),raw_mut)};
        
        if idx != ac {
            self.a = a2;
            self.b = b2;
            once(Matrix {
                a: a1,
                b: b1,
                d: cpy_d,
                asize: self.asize,
                bsize: self.bsize,
                dsize: self.dsize,
            })
        }
        else {
            self.a = a1;
            self.b = b2;
            once(Matrix {
                a: a2,
                b: b1,
                d: cpy_d,
                asize: self.asize,
                bsize: self.bsize,
                dsize: self.dsize,
            })
        }
    }
}


#[test]
fn test_mult_join() {
    let height = 2000;
    let width = 2000;
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let an = Array::from_shape_fn((height, width), |(i, j)| {
        (((j + i * width) % 3) as f32) - random
    });
    let bn = Array::from_shape_fn((width, height), |(i, j)| {
        (((j + 7 + i * height) % 3) as f32) + random
    });
    let mut dest = Array::zeros((height, height));
    let (ddim1,ddim2) = dest.dim();
    let m = Matrix {
        a: an.view(),
        b: bn.view(),
        d: dest.view_mut(),
        asize: an.dim(),
        bsize: bn.dim(),
        dsize: (ddim1,ddim2),
    };
    m.cut()
        .with_policy(Policy::Join(height * height / 32))
        .for_each(|e| {
            let a = e.a;
            let b = e.b;
            let mut output = e.d;
            linalg::general_mat_mul(1f32, &a, &b, 1f32, &mut output);
        });
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1f32, &an, &bn, 1f32, &mut verif);
    assert_abs_diff_eq!(
        dest.as_slice().unwrap(),
        verif.as_slice().unwrap(),
        epsilon = 1e-1f32
    );
}

#[test]
fn test_mult() {
    let height = 2000;
    let width = 2000;
    let mut rng = rand::thread_rng();
    let random = rng.gen_range(0.0, 1.0);
    let an = Array::from_shape_fn((height, width), |(i, j)| {
        (((j + i * width) % 3) as f32) - random
    });
    let bn = Array::from_shape_fn((width, height), |(i, j)| {
        (((j + 7 + i * height) % 3) as f32) + random
    });
    let mut dest = Array::zeros((height, height));
    let (ddim1,ddim2) = dest.dim();
    let m = Matrix {
        a: an.view(),
        b: bn.view(),
        d: dest.view_mut(),
        asize: an.dim(),
        bsize: bn.dim(),
        dsize: (ddim1,ddim2),
    };
    m.with_policy(Policy::Adaptive(20,2000))
        .for_each(|e| {
            let a = e.a;
            let b = e.b;
            let mut output = e.d;
            linalg::general_mat_mul(1f32, &a, &b, 1f32, &mut output);
        });
    let mut verif = Array::zeros((height, height));
    linalg::general_mat_mul(1f32, &an, &bn, 1f32, &mut verif);
    assert_abs_diff_eq!(
        dest.as_slice().unwrap(),
        verif.as_slice().unwrap(),
        epsilon = 1e-1f32
    );
}
