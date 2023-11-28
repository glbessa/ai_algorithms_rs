pub fn linear_search(vec: Vec<i32>, target: i32) -> usize {
    let mut i: usize = 0;

    while i < vec.len() {
        if vec[i] == target {
            break;
        }
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
    fn it_works() {

    }
}
