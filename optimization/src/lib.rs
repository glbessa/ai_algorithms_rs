use rand::Rng;
use std::io::prelude::*;
use std::fs::File;
use std::fmt;

pub fn factorial(mut n: usize) -> usize {
    let mut result = 1;
    loop {
        if n == 1 {
            break;
        }

        result *= n;
        n -= 1;
    }

    result
}

pub trait OptimizationProblem<T> {
    fn cost_function(&self) -> T;
    fn next_states(&self) -> Vec<Self> where Self: Sized;
    fn next_rand_state(&self) -> Self where Self: Sized;
}

pub trait State {

}

pub struct QueensState {
    board: Vec<Vec<bool>>
}

impl QueensState {
    pub fn new() -> Self {
        QueensState {
            board: Vec::new()
        }
    }

    pub fn new_random(num_queens: usize) -> Self {
        let mut board: Vec<Vec<bool>> = vec![vec![false; num_queens]; num_queens];
        let mut rng = rand::thread_rng();

        for i in 0..num_queens {
            let j = rng.gen_range(0..num_queens);

            board[i][j] = true;
        }

        QueensState {
            board: board
        }
    }

    pub fn from(board: Vec<Vec<bool>>) -> Self {
        QueensState {
            board: board
        }
    }

    pub fn from_file(file_name: &str) -> Result<Self, &'static str> {
        let mut f = File::open(file_name)
            .expect("Error while opening file!");

        let mut content: String = String::new();

        f.read_to_string(&mut content)
            .expect("Error while reading file!");

        let mut board: Vec<Vec<bool>> = Vec::new();
        let mut num_queens: usize = 0;
        let lines: Vec<&str> = content.split("\n").collect();
        
        for line in lines.into_iter() {
            let t: Vec<_> = line.split(" ")
                .collect();
            let mut t2: Vec<bool> = Vec::new();

            for col in t.into_iter() {
                if col == "1" {
                    t2.push(true);
                    num_queens += 1;
                }
                else {
                    t2.push(false);
                }
            }

            board.push(t2);
        }

        for v in board.clone().into_iter() {
            if v.len() != board.len() {
                return Err("Number of columns different from the number of lines!");
            }
        }

        if num_queens != board.len() {
            return Err("Number of queens different from the board dimension!");
        }

        Ok(QueensState::from(board))
    }
}

impl fmt::Display for QueensState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.board.len() {
            write!(f, "{} ", self.board.len() - i)?;
            for j in 0..self.board.len() {
                let mut text: &str = "-";
                
                if self.board[i][j] {
                    text = "♛";
                }
                
                if (j + i) % 2 == 0 {
                    print!("\x1b[37m");
                }
                write!(f, "{} ", text)?;
                if (j + i) % 2 == 0 {
                    print!("\x1b[0m");
                }
            }
            write!(f, "\n")?;
        }
        write!(f, "  A B C D E F G H")?;

        Ok(())
    }
}

impl OptimizationProblem<usize> for QueensState {
    fn cost_function(&self) -> usize {
        let mut cost: usize = 0;
        let mut n_queens_line: usize;
        let mut n_queens_col: usize;
        let mut n_queens_main_diag: usize;
        let mut n_queens_main_diag_sup: usize;
        let mut n_queens_inv_diag: usize;
        let mut n_queens_inv_diag_inf: usize;

        for i in 0..self.board.len(){
            n_queens_line = 0;
            n_queens_col = 0;
            n_queens_main_diag = 0;
            n_queens_main_diag_sup = 0;
            n_queens_inv_diag = 0;
            n_queens_inv_diag_inf = 0;

            for j in 0..self.board.len() {
                if self.board[i][j] {
                    n_queens_line += 1;
                }
                if self.board[j][i] {
                    n_queens_col += 1;
                }
                
                if self.board.len() > j + i && self.board[j+i][j] {
                    n_queens_main_diag += 1;
                }
                
                if self.board.len() > j + i + 1 && self.board[j][j+i + 1] {
                    n_queens_main_diag_sup += 1;
                }
                
                if self.board.len() > j + i && self.board[self.board.len() - j - i - 1][j] {
                    n_queens_inv_diag += 1;
                }

                if self.board.len() > j + i + 1  && self.board.len() > j + 1 && self.board[self.board.len() - j - 1][j + i + 1] {
                    n_queens_inv_diag_inf += 1;
                }
            }
            
            if n_queens_col >= 2 {
                cost += factorial(n_queens_col);
            }
            if n_queens_line >= 2 {
                cost += factorial(n_queens_line);
            }
            if n_queens_main_diag >= 2 {
                cost += factorial(n_queens_main_diag)
            }
            if n_queens_main_diag_sup >= 2 {
                cost += factorial(n_queens_main_diag_sup)
            }
            if n_queens_inv_diag >= 2 {
                cost += factorial(n_queens_inv_diag)
            }
            if n_queens_inv_diag_inf >= 2 {
                cost += factorial(n_queens_inv_diag_inf)
            }
        }

        cost
    }

    
    fn next_states(&self) -> Vec<Self> {
        let mut nx_st: Vec<QueensState> = Vec::new();
        
        for i in 0..self.board.len() {
            for j in 0..self.board.len() {
                if self.board[i][j] == false {
                    let mut new_board = self.board.clone();
                    new_board[i] = vec![false; self.board.len()];
                    new_board[i][j] = true;
                    nx_st.push(QueensState::from(new_board));
                }
            }
        }

        nx_st
    }

    fn next_rand_state(&self) -> Self {
        let mut new_board = self.board.clone();
        let mut rng = rand::thread_rng();

        let i: usize = rng.gen_range(0..self.board.len());
        let mut j: usize;
        loop {
            j = rng.gen_range(0..self.board.len());
            
            if self.board[i][j] == false {
                break;
            }
        }
        
        new_board[i] = vec![false; self.board.len()];
        new_board[i][j] = true;

        QueensState::from(new_board)
    }
}

pub fn hill_climbing(state: QueensState) -> QueensState {
    let mut actual_state: QueensState = state;
    let mut actual_cost: usize = actual_state.cost_function();
    let mut state_modified: bool = false;
    let mut neighbors: Vec<QueensState>;

    loop {
        neighbors = actual_state.next_states();

        for neig_idx in (0..neighbors.len()).rev() {
            if neighbors[neig_idx].cost_function() < actual_cost {
                actual_state = neighbors.remove(neig_idx);
                actual_cost = actual_state.cost_function();
                state_modified = true;
            }
        }

        if state_modified {
            state_modified = false;
        }
        else {
            break;
        }
    }

    actual_state
}

pub fn hill_climbing_with_slide_mov(state: QueensState) -> QueensState {
    let mut actual_state: QueensState = state;
    let mut actual_cost: usize = actual_state.cost_function();
    let mut state_modified: bool = false;
    let mut slided: bool = false;
    let mut neighbors: Vec<QueensState>;
    let mut max_rep: usize = 100;

    loop {
        neighbors = actual_state.next_states();

        for neig_idx in (0..neighbors.len()).rev() {
            if neighbors[neig_idx].cost_function() < actual_cost {
                actual_state = neighbors.remove(neig_idx);
                actual_cost = actual_state.cost_function();
                state_modified = true;
                slided = false;
            }
            else if neighbors[neig_idx].cost_function() == actual_cost {
                actual_state = neighbors.remove(neig_idx);
                actual_cost = actual_state.cost_function();
                state_modified = true;
                slided = true;
            }
        }

        if state_modified {
            state_modified = false;
        }
        else {
            break;
        }

        if slided {
            if max_rep > 0 {
                max_rep -= 1;
            }
            else {
                break;
            }
        }

        slided = false;
    }

    actual_state
}

pub fn simulated_annealing(state: QueensState, scheduler: fn(usize) -> f32) -> QueensState {
    let mut actual_state: QueensState = state;
    let mut iteration: usize = 1;
    let mut rng = rand::thread_rng();
    let mut next_state: QueensState;
    let mut var_cost: f32;
    let mut temperature: f32;

    loop {
        temperature = scheduler(iteration);

        if temperature <= 0.0 {
            return actual_state;
        }

        next_state = actual_state.next_rand_state();
        var_cost = (actual_state.cost_function()) as f32 - (next_state.cost_function()) as f32;
        
        if var_cost > 0.0 {
            actual_state = next_state;
        }
        else {
            let prob = (var_cost / temperature).exp();
            if rng.gen_range(0.0..1.0) <= prob {
                actual_state = next_state;
            }
        }
        iteration += 1;
    }
}

pub fn basic_scheduler(i: usize) -> f32 {
    10000.0 - (i as f32 * 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {

    }
}