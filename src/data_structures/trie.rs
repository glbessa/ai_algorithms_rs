use std::collections::HashMap;


// First implementation of a trie
struct Trie {
    terminal: bool,
    children: HashMap<char, Box<Trie>>
}



// Second implementation of a trie
struct Tst {
    terminal: bool,
    left: Option<Box<TstNode>>,
    center: Option<Box<TstNode>>,
    right: Option<Box<TstNode>>,
    value: Option<char>
}

impl<T> Tst<T> {
    pub fn new() -> Self {
        Tst {
            left: None,
            center: None,
            right: None,
            terminal: false,
            value: None
        }
    }

    pub fn insert(&mut self, data: &String) {
        todo!();        
    }

    pub fn remove(&mut self, data: &String) {
        todo!();
    }

    pub fn contains(&self, data: &String) -> bool {
        todo!();
    }

    pub fn search(&self, data: &String) -> Vec<String> {
        todo!();
    }
}