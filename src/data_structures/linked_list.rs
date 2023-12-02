use std::mem;

#[derive(Debug)]
pub struct LinkedList<T> {
    value: Option<T>,
    next: Option<Box<LinkedList<T>>>
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        return LinkedList{
            value: None,
            next: None
        };
    }

    pub fn insert(&mut self, index: usize, data: T) {
        let mut actual = self;
        let mut idx: usize = 0;

        loop {
            if idx == index {
                actual.value = Some(data);
                break;
            }
            if actual.next.is_none() {
                actual .next = Some(Box::new(LinkedList{
                    value: Some(data),
                    next: None
                }));
                break;
            }
            if let Some(ref mut next) = actual.next {
                actual = &mut *next;
            }

            idx += 1;
        }
    }

    pub fn push(&mut self, data: T) {
        let mut actual = self;

        if actual.value.is_none() {
            actual.value = Some(data);
            return;
        }

        loop {
            if actual.next.is_none() {
                actual.next = Some(Box::new(LinkedList {
                    value: Some(data),
                    next: None
                }));
                return;
            }

            if let Some(ref mut nxt) = actual.next {
                actual = &mut *nxt;
            }
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let mut actual = self;
        let value: Option<T>;

        value = actual.value.take();

        if let Some(ref mut nxt) = actual.next {
            actual.value = nxt.value.take();
            actual.next = nxt.next.take();
        }

        return value;
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        let mut idx = 0;
        let mut actual = self;
        let value: Option<T>;

        if index == 0 {
            return actual.pop();
        }

        loop {
            if actual.next.is_none() {
                return None;
            }

            if idx + 1 == index {
                value = actual.next.unwrap().value.take();
                actual.next = actual.next.unwrap().next.take();
                    
                return value;
            }


            actual = &mut *actual.next.unwrap();
        }

            idx += 1;
        
        
        None
    }
}

#[cfg(tests)]
mod tests {
    use super::*;

    #[test]
    fn inserting_at_begining() {
        let mut ll = LinkedList::new();
        ll.insert(0, 20);

    }

    #[test]
    fn inserting_at_end() {

    }

    #[test]
    fn pushing() {

    }

    #[test]
    fn poping() {

    }
}

fn main() {
    let mut ll = LinkedList::new();
    ll.push(10);
    ll.push(30);
    //ll.push(40);
    println!("{}", ll.pop().unwrap_or(0));
    println!("{}", ll.pop().unwrap_or(0));
    println!("{}", ll.pop().unwrap_or(0));

    println!("{:?}", ll);
}