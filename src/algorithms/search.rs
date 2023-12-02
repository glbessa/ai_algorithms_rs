pub fn linear_search(vec: Vec<i32>, target: i32) -> usize {
    let mut i: usize = 0;

    while i < vec.len() {
        if vec[i] == target {
            break;
        }
        i += 1;
    }

    i
}

pub fn binary_search(vec: Vec<i32>, target: i32) -> usize {
    let mut i: usize = vec.len() / 2;

    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_test() {
        let v = vec![5, 7, 2, 9, 1, 0, 3, 6, 2];
        let result = linear_search(v, 9);

        assert_eq!(result, 3);
    }
}
