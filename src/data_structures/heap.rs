use std::cmp::PartialOrd;

pub struct Heap<T: PartialOrd, V> (Option<(T, V, Box<Heap<T, V>>)>);

impl<T: PartialOrd, V> Heap<T, V> {
    pub fn new() -> Self {
        Heap(None)
    }

    pub fn push(&mut self, weight: T, data: V) {
        match self.0 {
            Some((wei, _, ref mut next)) => {
                if wei < weight {
                    Some((weight, data, Box::new(Heap(self.0))))
                }
                else {
                    next.push(weight, data);
                }
            },
            None => {
                Some((weight, data, Box::new(Heap(None))))
            }
        }
        /*
        self.0 = match self.0 {
            Some((wei, dat, ref mut next)) => {
                if wei < weight {
                    next.push(weight, data);
                    self.0
                }
                else {
                    //let t = Heap(Some((wei, dat, next)));
                    Some((weight, data, Box::new(Heap(Some((wei, dat, next))))))
                }
            },
            None => Some((weight, data, Box::new(Heap(None))))
        }
        */
    }
}

fn main() {

}