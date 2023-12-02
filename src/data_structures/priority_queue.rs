
use std::iter::Iterator;

pub struct PriorityQueue<T> {
    priority: Vec<i32>,
    queue: Vec<T>
}

impl<T> PriorityQueue<T> {
    pub fn new() -> Self {
        PriorityQueue { 
            priority: Vec::new(),
            queue: Vec::new()
        }
    }

    pub fn put(&mut self, item_priority:i32, item: T) {
        if self.priority.len() == 0 {
            self.priority.push(item_priority);
            self.queue.push(item);
            return ();
        }

        if self.priority[0] <= item_priority {
            self.priority.insert(0, item_priority);
            self.queue.insert(0, item);
            return ();
        }

        for priority_idx in (0..self.priority.len()).rev() {
            if self.priority[priority_idx] <= item_priority {
                self.priority.insert(priority_idx, item_priority);
                self.queue.insert(priority_idx, item);
                return ();
            }
        }

        self.priority.push(item_priority);
        self.queue.push(item);
    }

    pub fn pop(&mut self) -> Option<(i32, T)> {
        let prio = match self.priority.pop() {
            Some(v) => v,
            None => return None
        };
        let value = match self.queue.pop() {
            Some(v) => v,
            None => return None
        };

        return Some((prio, value));
    }

    pub fn len(&self) -> usize {
        return self.queue.len();
    }
}

impl<T> Iterator for PriorityQueue<T> {
    type Item = (i32, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.pop()
    }
}
