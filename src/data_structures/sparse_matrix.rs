pub struct SparseMatrix<T: fmt::Display + From<usize> + Copy> {
    head: Option<Box<SparseMatrixLine<T>>>
}

struct SparseMatrixLine<T: fmt::Display + From<usize> + Copy> {
    below : Option<Box<SparseMatrixLine<T>>>,
    right: Option<Box<SparseMatrixCell<T>>>,
    line: usize,
}

struct SparseMatrixCell<T: fmt::Display + From<usize> + Copy> {
    right: Option<Box<SparseMatrixCell<T>>>,
    column: usize,
    value: T
}

impl<T: fmt::Display + From<usize> + Copy> SparseMatrix<T> {
    pub fn new() -> Self {
        SparseMatrix { head: None }
    }

    pub fn get_element(&self, line: usize, column: usize) -> T {
        match self.head {
            Some(ref next) => { return next.get_element(line, column); },
            None => { return T::from(0); }
        }
    }

    pub fn set_element(&mut self, line: usize, column: usize, data: T) {
        match self.head {
            Some(ref mut next) => {
                next.set_element(line, column, data)
            },
            None => {
                self.head = Some(Box::new(SparseMatrixLine { below: None, right: None, line: line }));
                self.set_element(line, column, data);
            }
        }
    }

    pub fn add(&self, mat2: &Self) -> Self {
        let mut new_mat: SparseMatrix<T> = SparseMatrix::new();
        todo!();
    }

    pub fn subtract(&self, mat2: &Self) -> Self {
        todo!();
    }

    pub fn multiply(&self, mat2: &Self) -> Self {
        todo!();
    }
}

impl<T: fmt::Display + From<usize> + Copy> SparseMatrixLine<T> {
    fn get_element(&self, line: usize, column: usize) -> T {
        if line == self.line {
            match self.right {
                Some(ref next) => { return next.get_element(column); },
                None => { return T::from(0); }
            }
        }
        else if line > self.line {
            return T::from(0);
        }
        match self.below {
            Some(ref next) => { return next.get_element(line, column); },
            None => { return T::from(0); }
        }
    }

    fn set_element(&mut self, line: usize, column: usize, data: T) {
        if line == self.line {
            match self.right {
                Some(ref mut next) => {
                    next.set_element(column, data)
                },
                None => {
                    self.right = Some(Box::new(SparseMatrixCell { right: None, column: column, value: data }))
                }
            }
        }
        else if line > self.line {
            let t = self.below.take();
            let t2 = self.right.take();
            self.below = Some(Box::new( SparseMatrixLine { below: t, right: t2, line: self.line }));
            self.line = line;
            self.right = Some(Box::new(SparseMatrixCell { right: None, column: column, value: data }));
        }
    }
}

impl<T: fmt::Display + From<usize> + Copy> SparseMatrixCell<T> {
    fn get_element(&self, column: usize) -> T {
        if column == self.column {
            return self.value;
        }
        else if column > self.column {
            return T::from(0);
        }
        match self.right {
            Some(ref next) => { return next.get_element(column); },
            None => { return T::from(0); }
        }
    }

    fn set_element(&mut self, column: usize, data: T) {
        if column == self.column {
            self.value = data;
        }
        else if column > self.column {
            let t = self.right.take();
            self.right = Some(Box::new(SparseMatrixCell { right: t, column: self.column, value: self.value }));
            self.column = column;
            self.value = data;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_element_test() {
        let mut m = SparseMatrix::<usize>::new();

        m.set_element(10, 20, 59);
        let t = m.get_element(10, 20);

        assert_eq!(t, 59);
    }

    #[test]
    fn get_element_0() {
        let m = SparseMatrix::<usize>::new();

        let t = m.get_element(10, 20);

        assert_eq!(t, 0);
    }

    #[test]
    fn set_element_test() {
        let mut m = SparseMatrix::<usize>::new();

        m.set_element(10, 20, 59);
        let t = m.get_element(10, 20);

        assert_eq!(t, 59);
    }
}
