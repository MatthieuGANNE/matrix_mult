use rayon_adaptive::prelude::*;
use rayon_adaptive::BasicPower;
use std::iter::{once, Once};
///
pub struct Split<D, S, L>
where
    D: Send,
    S: Fn(D) -> (D, D) + Sync + Send + Clone,
    L: Fn(&D) -> usize + Clone + Send,
{
    pub data: D,
    pub splitter: S,
    pub length: L,
}

impl<D, S, L> Divisible for Split<D, S, L>
where
    D: Send,
    S: Fn(D) -> (D, D) + Sync + Send + Clone,
    L: Fn(&D) -> usize + Clone + Send,
{
    type Power = BasicPower;
    fn base_length(&self) -> Option<usize> {
        Some((self.length)(&self.data))
    }

    #[allow(unused_variables)]
    fn divide_at(self, index: usize) -> (Self, Self) {
        let (d1, d2) = (self.splitter)(self.data);
        (
            Split {
                data: d1,
                splitter: self.splitter.clone(),
                length: self.length.clone(),
            },
            Split {
                data: d2,
                splitter: self.splitter,
                length: self.length,
            },
        )
    }
}

impl<D, S, L> ParallelIterator for Split<D, S, L>
where
    D: Send,
    S: Fn(D) -> (D, D) + Sync + Send + Clone,
    L: Fn(&D) -> usize + Clone + Send,
{
    type Item = D;

    type SequentialIterator = Once<D>;

    fn to_sequential(self) -> Self::SequentialIterator {
        once(self.data)
    }

    #[allow(unused_variables)]
    fn extract_iter(&mut self, size: usize) -> Self::SequentialIterator {
        panic!("extract_iter");
    }
}

pub fn split<D, S, L>(data: D, splitter: S, length: L) -> Split<D, S, L>
where
    D: Send,
    S: Fn(D) -> (D, D) + Sync + Send + Clone,
    L: Fn(&D) -> usize + Clone + Send,
{
    Split {
        data: data,
        splitter: splitter,
        length: length,
    }
}
