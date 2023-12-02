


fn main() {
    let profits = vec![1, 2, 5, 6];
    let weights = vec![2, 3, 4, 5];
    println!("{:?}", knapsack_0_1(&profits, &weights, 8));
}

fn unbounded_knapsack(profits: &Vec<i32>, weights: &Vec<usize>) {

}

fn knapsack_0_1(profits: &Vec<usize>, weights: &Vec<usize>, limit: usize) -> Vec<usize> {
    let mut lookup_table: Vec<Vec<usize>> = vec![vec![0; limit + 1]; profits.len() + 1];
    let mut selection: Vec<usize> = Vec::new();

    for i in 1..=profits.len() {
        for j in 1..=limit {
            if j >= weights[i - 1] && lookup_table[i - 1][j] < lookup_table[i - 1][j - weights[i - 1]] + profits[i - 1] {
                lookup_table[i][j] = lookup_table[i - 1][j - weights[i - 1]] + profits[i - 1];
            } else {
                lookup_table[i][j] = lookup_table[i - 1][j];
            }
        }
    }

    let mut aux: usize = lookup_table[profits.len()][limit];
    let mut flag: bool;
    for i in (1..profits.len()).rev() {
        println!("{}", aux);
        flag = false;
        for j in (1..=limit).rev() {
            if lookup_table[i][j] < aux {
                break;
            } else if lookup_table[i][j] == aux {
                flag = true;
                break;
            }
        }

        if !flag {
            aux -= weights[i];
            selection.push(i);
        }
    }

    selection
}