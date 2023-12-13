// https://doc.rust-lang.org/stable/std/collections/struct.BinaryHeap.html
use std::collections::{LinkedList, HashSet, BinaryHeap, VecDeque};
use std::cmp::{Ordering, Reverse, Eq, Ord}; // https://doc.rust-lang.org/std/cmp/index.html
use std::clone::Clone;
use std::fs::File;
use std::io::prelude::*;
use std::process::exit;
use std::convert::TryInto;
use std::fmt;

// Edge Struct Declaration -------------
pub struct Edge {
    src: u32,
    dst: u32,
    weight: i32
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.src == other.src && self.dst == other.dst
    }
}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.weight.cmp(&other.weight))
    }
}

impl Eq for Edge {
    fn assert_receiver_is_total_eq(&self) {

    }
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight.cmp(&other.weight)
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let _ = write!(f, "Edge (src: {}; dst: {}; weight: {})\n", self.src, self.dst, self.weight)?;
        Ok(())
    }
}
// -------------------------------
/*
pub trait AdjMatrix<T> {
    fn insert_vertex(&mut self, vertex: V);
    fn remove_vertex(&mut self, vertex_index: usize) -> Result<(), &'static str>;
    fn insert_edge(&mut self, src_idx: usize, dst_idx: usize, edge_weight: u64, directed: bool) -> Result<(), &'static str>;
    fn remove_edge(&mut self, src_idx: usize, dst_idx: usize, directed: bool) -> Result<(), &'static str>;
    fn get_edge_weight(&self, src_idx: usize, dst_idx: usize) -> Result<T, &'static str>;
    fn get_adjacent_vertices(&self, vertex_idx: usize) -> Result<Vec<usize>, &'static str>;
}
*/
// Generics: https://doc.rust-lang.org/book/ch10-01-syntax.html
#[derive(Debug)]
pub struct Graph<V: Eq + fmt::Display + Clone, M: AdjMatrix<u32> + Clone + fmt::Display, I> {
    vertices: Vec<V>,
    //relations: Vec<Vec<u64>>
    relations: u64
}

// https://doc.rust-lang.org/rust-by-example/hello/print/print_display.html
impl<V: Eq + fmt::Display + Clone> fmt::Display for Graph<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Graph (\n");
        for i in 0..self.relations.len() {
            for j in 0..self.relations.len() {
                if self.get_edge_weight(i, j).unwrap() != 0 {
                    write!(f, "\t{} -- \t{}\t --> {}\n", self.get_vertex(i).unwrap(), self.get_edge_weight(i, j).unwrap(), self.get_vertex(j).unwrap());
                }
            }
        }
        write!(f, ")\n");

        Ok(())
    }
}

impl Graph<String> {
    pub fn from_file(file_name: String) -> Result<Self, &'static str> {
        let mut f = File::open(file_name)
            .expect("Error while opening file!");

        let mut content: String = String::new();
        f.read_to_string(&mut content)
            .expect("Error while reading file!");

        let vertices_ids: Vec<String>;
        let mut adjacency_matrix: Vec<Vec<u64>> = Vec::new();
        let mut lines: Vec<&str> = content.split("\n").collect();

        vertices_ids = lines.remove(0).split(",").map(|s| String::from(s)).collect();
        
        for line in lines.into_iter() {
            let t: Vec<_> = line.split(" ")
                .map(|s| s.parse::<u64>())
                .map(Result::unwrap)
                .collect();

            adjacency_matrix.push(t);
        }

        Ok(Graph::from(vertices_ids, adjacency_matrix))        
    }

    pub fn with_n_vertices(n_vertices: usize) -> Self {
        let mut vertices: Vec<String> = vec![String::from(""); n_vertices];
        let mut adjacency_matrix: Vec<Vec<u64>> = vec![vec![0; n_vertices]; n_vertices];

        Graph::from(vertices, adjacency_matrix)
    }
}

impl<V: Eq + fmt::Display + Clone> Graph<V> {
    pub fn new() -> Self {
        return Graph {
            vertices: Vec::new(),
            relations: Vec::new()
        };
    }

    pub fn from(vertices_ids: Vec<V>, adjacency_matrix: Vec<Vec<u64>>) -> Self {
        Graph {
            vertices: vertices_ids,
            relations: adjacency_matrix
        }
    }

    pub fn from_only_with_vertices(vertices_ids: Vec<V>) -> Self {
        let mut adjacency_matrix: Vec<Vec<u64>> = vec![vec![0; vertices_ids.len()]; vertices_ids.len()];

        Graph::from(vertices_ids, adjacency_matrix)
    }

    pub fn to_file(&self, file_name: String) -> Result<(), &'static str> {


        Ok(())
    }

    pub fn insert_vertex(&mut self, vertex: V) {
        self.vertices.push(vertex);

        self.relations.push(vec![0; self.vertices.len()]);

        for i in 0..self.relations.len() {
            self.relations[i].push(0);
        }
    }

    pub fn remove_vertex(&mut self, vertex_index: usize) -> Result<(), &'static str> {
        if self.vertices.len() <= vertex_index {
            return Err("Index out of range!");
        }

        self.vertices.remove(vertex_index);

        Ok(())
    }

    pub fn get_vertex(&self, vertex_idx: usize) -> Result<&V, &'static str> {
        if self.vertices.len() <= vertex_idx {
            return Err("Index out of range!");
        }

        Ok(&self.vertices[vertex_idx])
    }

    pub fn insert_edge(&mut self, src_idx: usize, dst_idx: usize, edge_weight: u64, directed: bool) -> Result<(), &'static str> {
        if self.vertices.len() <= src_idx || self.vertices.len() <= dst_idx {
            return Err("Index out of range!");
        }

        self.relations[src_idx][dst_idx] = edge_weight;
        
        if directed == true {
            self.relations[dst_idx][src_idx] = edge_weight;
        }

        Ok(())
    }

    pub fn remove_edge(&mut self, src_idx: usize, dst_idx: usize, directed: bool) -> Result<(), &'static str> {
        if self.num_vertices() <= src_idx || self.num_vertices() <= dst_idx {
            return Err("Index out of range!");
        }

        self.relations[src_idx][dst_idx] = 0;
        
        if directed == true {
            self.relations[dst_idx][src_idx] = 0;
        }        

        Ok(())
    }

    pub fn get_edge_weight(&self, src_idx: usize, dst_idx: usize) -> Result<u64, &'static str> {
        if self.num_vertices() <= src_idx || self.num_vertices() <= dst_idx {
            return Err("Index out of range!");
        }

        let weight: u64 = self.relations[src_idx][dst_idx];

        Ok(weight)
    }

    pub fn get_adjacent_vertices(&self, vertex_idx: usize) -> Result<Vec<usize>, &'static str> {
        if self.num_vertices() <= vertex_idx {
            return Err("Index out of range!");
        }

        let mut adjacent_vertices: Vec<usize> = Vec::new();
        
        for i in 0..self.num_vertices() {
            if self.get_edge_weight(vertex_idx, i).unwrap() > 0 {
                adjacent_vertices.push(i);
            }
        }

        Ok(adjacent_vertices)
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }
    
    pub fn breadth_first_search(&self, src: usize, dst: usize) -> Result<Vec<usize>, &'static str> {
        let mut visited_vertices: Vec<bool> = vec![false; self.vertices.len()];
        let mut frontier: VecDeque<usize> = VecDeque::new();
        let mut path: Vec<usize> = Vec::new();
        let mut possible_path: Vec<(usize, usize)> = Vec::new();
        let mut parent: usize;
        let mut actual_node: usize;

        frontier.push_back(actual_node);

        loop {
            if frontier.len() == 0 {
                return Err("No path available between the points informed!");
            }

            parent = actual_node;
            actual_node = frontier.pop_front();
            visited_vertices[actual_node] = true;
            possible_path.push((parent, actual_node));

            if actual_node == dst {
                break;
            }

            for i in 0..self.num_vertices() {
                if self.get_edge_weight(actual_node, i) != 0 && visited_vertices[i] == false {
                    frontier.push_back(i);
                }
            }
        }

        actual_node = possible_path.pop();

        while actual_node != src {
            parent = possible_path.pop();
            
            if actual_node.0 == parent.1 {
                path.push(actual_node);
                actual_node = parent.0;
            }
        }

        path.push(src);

        Ok(path.reverse())
    }

    pub fn depth_first_search(&self, src: usize, dst: usize) -> Vec<usize> {
        let mut visited_vertices: Vec<bool> = vec![false; self.vertices.len()];
        let mut frontier: Vec<usize> = Vec::new();
        let mut path: Vec<usize> = Vec::new();
        let mut possible_path: Vec<(usize, usize)> = Vec::new();
        let mut parent: usize;
        let mut actual_node: usize;

        frontier.push(actual_node);

        loop {
            if frontier.len() == 0 {
                return Err("No path available between the points informed!");
            }

            parent = actual_node;
            actual_node = frontier.pop();
            visited_vertices[actual_node] = true;
            possible_path.push((parent, actual_node));

            if actual_node == dst {
                break;
            }

            for i in 0..self.num_vertices() {
                if self.get_edge_weight(actual_node, i) != 0 && visited_vertices[i] == false {
                    frontier.push(i);
                }
            }
        }

        actual_node = possible_path.pop();

        while actual_node != src {
            parent = possible_path.pop();
            
            if actual_node.0 == parent.1 {
                path.push(actual_node);
                actual_node = parent.0;
            }
        }

        path.push(src);

        Ok(path.reverse())
    }
    
    pub fn depth_limited_search(&self, src: usize, dst: usize) -> Vec<usize> {
        Vec::new()
    }
    
    pub fn iterative_deepening_search(&self, src: usize, dst: usize) -> Vec<usize> {
        Vec::new()
    }
    
    pub fn best_first_search(&self, src: usize, dst: usize) -> Vec<usize> {
        Vec::new()
    }
    
    pub fn greedy_best_first_search(&self, src: usize, dst: usize) -> Vec<usize> {
        Vec::new()
    }
    
    pub fn ida_star(&self, src: usize, dst: usize) -> Vec<usize> {
        Vec::new()
    }
    
    pub fn sma_star(&self, src: usize, dst: usize) -> Vec<usize> {
        Vec::new()
    }
    
    pub fn a_star(&self, src: usize, dst: usize, heuristic_dist: Vec<u64>) -> Vec<usize> {
        Vec::new()
    }
    
    pub fn hill_climbing(&self, cost_func: fn(usize) -> usize) -> Self {
        Graph::new()
    }
    
    pub fn random_restart_hill_climbing(&self) -> Self {
        Graph::new()
    }
    
    pub fn simulated_annealing(&self) -> Self {
        Graph::new()
    }
    
    pub fn local_beam_search(&self) -> Self {
        Graph::new()
    }
    
    pub fn genetic_algorithms(&self) -> Self {
        Graph::new()
    }

    // Algoritmo de Dijkstra
    pub fn get_dijkstra_path(&self, src_idx: usize, dst_idx: usize) -> Result<VecDeque<usize>, &'static str> {
        if self.vertices.len() <= src_idx || self.vertices.len() <= dst_idx {
            return Err("Index out of range!");
        }

        let mut previous_vertex: Vec<Option<usize>> = vec![None; self.vertices.len()];
        let mut path_cost: Vec<Option<usize>> = vec![None; self.vertices.len()];
        let mut is_closed: Vec<bool> = vec![false; self.vertices.len()];
        let mut vert_to_visit: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
        let mut vertex_idx: usize;

        // custo do no inicial é 0
        path_cost[src_idx] = Some(0);
        // insere o no inicial na heap
        vert_to_visit.push(Reverse((0, src_idx)));

        loop {
            // remove o primeiro nó da heap
            vertex_idx = match vert_to_visit.pop() {
                Some(Reverse((_, v))) => v,
                None => break
            };

            // verifica se esse no ja nao esta fechado
            if is_closed[vertex_idx] {
                continue;
            }

            // fecha o no
            is_closed[vertex_idx] = true;

            // verifica os adjacentes desse no, atualiza seus pesos e adiciona na heap
            match self.get_adjacent_vertices(vertex_idx) {
                Ok(list) => {
                    for idx in list.iter() {
                        if is_closed[*idx] {
                            continue;
                        }

                        if self.get_edge_weight(vertex_idx, *idx).unwrap() < 0 {
                            continue;
                        }

                        let total_cost: usize = match path_cost[vertex_idx] {
                            Some(cost_v_idx) => cost_v_idx + self.get_edge_weight(vertex_idx, *idx).unwrap() as usize,
                            None => self.get_edge_weight(vertex_idx, *idx).unwrap() as usize
                        };

                        match path_cost[*idx] {
                            Some(cost) => {
                                if cost > total_cost {
                                    path_cost[*idx] = Some(total_cost);
                                    previous_vertex[*idx] = Some(vertex_idx);
                                    vert_to_visit.push(Reverse((total_cost.try_into().unwrap(), *idx)));
                                }
                            },
                            None => {
                                path_cost[*idx] = Some(total_cost);
                                previous_vertex[*idx] = Some(vertex_idx);
                                vert_to_visit.push(Reverse((total_cost.try_into().unwrap(), *idx)));
                            }
                        }
                    }
                },
                Err(msg) => return Err(msg)
            }
        }

        let mut path: VecDeque<usize> = VecDeque::new();
        path.push_front(dst_idx);
        let mut actual_vertex: usize = dst_idx;

        loop {
            actual_vertex = match previous_vertex[actual_vertex] {
                Some(v_idx) => v_idx,
                None => break
            };

            path.push_front(actual_vertex);
        }

        Ok(path)
    }

    // Algoritmo de Kruskal: o conjunto A é  uma floresta cujos vértices são todos os vértices do grafo e a aresta segura
    //   adicionada é sempre uma aresta de peso mínimo no grafo que conecta duas componentes distintas.
    pub fn get_mst_kruskal(&self) -> Self {
        let mut a: HashSet<(usize, usize)> = HashSet::new();
        let mut v_sets: Vec<HashSet<usize>> = Vec::new();
        let mut heap: BinaryHeap<Reverse<(u64, (usize, usize))>> = BinaryHeap::new();

        // Criando a floresta de conjuntos
        for i in 0..self.vertices.len() {
            v_sets.push(HashSet::from([i]));
        }

        // Ordenando as arestas
        for i in 0..self.relations.len() {
            for j in 0..self.relations.len() {
                if self.get_edge_weight(i, j).unwrap() != 0 {
                    heap.push(Reverse((self.get_edge_weight(i, j).unwrap(), (i, j))));
                }
            }
        }

        for _ in 0..heap.len() {
            // Remove a aresta de menor peso da heap
            let (u, v) = match heap.pop() {
                Some(Reverse((_, (u1, v1)))) => (u1, v1),
                None => (0, 0)
            };
            let (mut set1_idx, mut set2_idx): (usize, usize) = (0, 0);

            // Verifica em todos os sets se src ou dst estão inclusos neles
            for i in 0..v_sets.len() {
                if v_sets[i].contains(&u) {
                    set1_idx = i;
                }

                if v_sets[i].contains(&v) {
                    set2_idx = i;
                }
            }

            // Caso os dois não estejam no mesmo set
            if set1_idx != set2_idx {
                // Insere-se essa aresta
                a.insert((u, v));
                
                // E junta os sets
                let t = v_sets.remove(set2_idx);
                
                if set2_idx < set1_idx {
                    v_sets[set1_idx-1].extend(&t);
                }
                else {
                    v_sets[set1_idx].extend(&t);
                }
            }
        }

        let mut adjacency_matrix: Vec<Vec<u64>> = vec![vec![0; self.relations.len()]; self.relations.len()];
        for (src, dst) in a.into_iter() {
            adjacency_matrix[src][dst] = self.get_edge_weight(src, dst).unwrap();
        }

        return Graph::from(self.vertices.clone(), adjacency_matrix);
    }

    // Algoritmo de Prim: inicia adicionando ao conjunto A os vertices ligados pela aresta de menor custo
    //      e após vai adicionando os vertices que tiverem menor custo e estejam sejam adjacentes aos já existentes
    // https://pt.wikipedia.org/wiki/Algoritmo_de_Prim
    pub fn get_mst_prim(&self) -> Self {
        let mut a: HashSet<usize> = HashSet::with_capacity(self.vertices.len());
        let mut heap: BinaryHeap<Reverse<(u64, (usize, usize))>> = BinaryHeap::new();
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(self.vertices.len() - 1);

        // Pega a aresta de menor valor diferente de zero
        let mut min: (usize, usize) = (0, 0);
        for i in 0..self.vertices.len() {
            for j in 0..self.vertices.len() {
                if self.get_edge_weight(i, j).unwrap() != 0 && self.get_edge_weight(i, j).unwrap() < self.get_edge_weight(min.0, min.1).unwrap() {
                    min = (i, j);
                }
            }
        }

        // Adicionando a primeira aresta na heap
        heap.push(Reverse((self.get_edge_weight(min.0, min.1).unwrap(), min)));
        a.insert(min.0);
        let adjacents = self.get_adjacent_vertices(min.0).unwrap();

        // Adicionando as arestas para adjacentes do primeiro vertice na heap ---
        for adj_vertex in adjacents.into_iter() {
            if a.contains(&adj_vertex) {
                continue;
            }

            heap.push(Reverse((self.get_edge_weight(min.0, adj_vertex).unwrap(), (min.0, adj_vertex))));
        }

        // Itera ate que todos os vertices estejam acessiveis
        loop {
            if a.len() == self.vertices.len() {
                break;
            }

            // Remove da heap
            let (priority, src, dst) = match heap.pop() {
                Some(Reverse((prio, (sr, ds)))) => (prio, sr, ds),
                None => { continue; }
            };

            if a.contains(&dst) {
                continue;
            }

            // Insere um vertice que nao havia sido explorado ainda
            a.insert(dst);
            // Adiciona a aresta no vetor de arestas
            edges.push((src, dst));
            // Pega os adjacentes
            let adjacents = self.get_adjacent_vertices(dst).unwrap();

            for adj_vertex in adjacents.into_iter() {
                if a.contains(&adj_vertex) {
                    continue;
                }

                // Popula a heap denovo
                heap.push(Reverse((self.get_edge_weight(dst, adj_vertex).unwrap(), (dst, adj_vertex))));
            }
        }

        // Transforma tudo em um novo grafo :)
        let mut adjacency_matrix: Vec<Vec<u64>> = vec![vec![0; self.relations.len()]; self.relations.len()];
        for (src, dst) in edges.into_iter() {
            adjacency_matrix[src][dst] = self.get_edge_weight(src, dst).unwrap();
        }

        return Graph::from(self.vertices.clone(), adjacency_matrix);
    }

    // Algoritmo de Boruvka:
    // https://pt.wikipedia.org/wiki/Algoritmo_de_Bor%C5%AFvka
    pub fn get_mst_boruvka(&self) -> Self {
        let mut vertices: Vec<V> = self.vertices.clone();
        let mut adjacency_matrix: Vec<Vec<u64>> = vec![vec![0; self.relations.len()]; self.relations.len()];
        let mut graph: Graph<V> = Graph::from(vertices, adjacency_matrix);

        todo!();
    }

    pub fn get_random_route(&self) -> Result<Vec<usize>, &'static str> {
        todo!()
    }

    pub fn get_route_cost(&self, route: &Vec<usize>) -> Result<u64, &'static str> {
        if !self.is_squared() {
            return Err("Graph is not squared!");
        }

        let mut cost: u64 = 0;

        for i in 0..route.len() - 1 {
            cost += self.get_edge_weight(route[i], route[i+1])?;
        }

        cost += self.get_edge_weight(route[route.len()-1], route[0])?;

        Ok(cost)
    }

    pub fn get_route_cost_2(&self, route: &Vec<&usize>) -> Result<u64, &'static str> {
        if !self.is_squared() {
            return Err("Graph is not squared!");
        }

        let mut cost: u64 = 0;

        for i in 0..route.len() - 1 {
            cost += self.get_edge_weight(*route[i], *route[i+1])?;
        }

        cost += self.get_edge_weight(*route[route.len()-1], *route[0])?;

        Ok(cost)
    }

    // Fleury algorithm: https://en.wikipedia.org/wiki/Eulerian_path#Fleury's_algorithm
    pub fn get_eulerian_path(&self) -> Result<Vec<usize>, &'static str> {
        let mut eulerian_path: Vec<usize> = Vec::new();

        for i in 0..self.num_vertices() {
            if self.get_adjacent_vertices(i)?.len() % 2 != 0 {
                return Err("Graph does not have an eulerian path!");
            }
        }


        
    }

    pub fn perfect_matching(&self) -> Result<Self, &'static str> {
        let mut visited_vertices = vec![false; self.num_vertices()];
        let mut max_coup_graph = Graph::new();

        for i in 0..self.num_vertices() {
            if visited_vertices[i] {
                continue;
            }

            visited_vertices[i] = true;
            let mut min_cost_vertex: usize = 0;

            for j in 0..self.num_vertices() {
                if visited_vertices[j] {
                    continue;
                }

                if self.get_edge_weight(i, j)? < min_cost {
                    min_cost = j;
                }
            }

            visited_vertices[min_cost_vertex] = true;
            max_coup_graph.insert_vertex(self.get_vertex(i)?.clone());
            max_coup_graph.insert_vertex(self.get_vertex(min_cost_vertex)?.clone());
            max_coup_graph.insert_edge(i, min_cost_vertex, self.get_edge_weight(i, min_cost_vertex)?, false);
        }

        Ok(max_coup_graph)
    }

    pub fn get_odd_degree_vertices(&self) -> Result<Vec<usize>, &'static str> {
        let mut odd_degree_vertices: Vec<usize> = Vec::new();

        for i in 0..self.num_vertices() {
            if self.get_adjacent_vertices(i)?.len() % 2 != 0 {
                odd_degree_vertices.push(i);
            }
        }

        Ok(odd_degree_vertices)
    }

    pub fn tsp_brute_force(&self, log: bool) -> Result<(Vec<usize>, u64), &'static str> {
        if !self.is_squared() {
            return Err("Graph is not squared!");
        }

        let first_route: Vec<usize> = (0..self.num_vertices()).collect();
        let mut actual_cost: u64 = self.get_route_cost(&first_route)?;
        let mut best_route: Vec<usize> = first_route.clone();
        let mut best_cost: u64 = actual_cost.clone();
        let mut counter: usize = 0;

        let start_time = Instant::now();

        let permutations = first_route.iter().permutations(self.num_vertices() - 1);

        for permutation in permutations {
            counter += 1;
            permutation.push(&(num_vertices - 1));
            actual_cost = self.get_route_cost_2(&permutation)?;

            if actual_cost < best_cost {
                best_cost = actual_cost.clone();
                best_route = permutation.into_iter().cloned().collect();
            }

            if log && counter % 10000 == 0{
                println!("Iteration: {} - Time elapsed: {} - Route cost: {}", counter, Instant::now().duration_since(start_time).as_micros(), best_cost);
            }
        }
        
        let end_time = Instant::now();

        if log {
            println!("Total iterations: {} - Time elapsed: {} - Best route cost: {} - Best route found: {}", counter, end_time.duration_since(start_time).as_micros(), best_cost, best_route.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" -> "));
        }

        Ok((best_route, best_cost))
    }

    pub fn tsp_2_opt_approx(&self, log: bool) -> Result<(Vec<usize>, u64), &'static str> {
        if !self.is_squared() {
            return Err("Graph is not squared!");
        }

        let mut actual_slice: Vec<usize>;
        let mut actual_route: Vec<usize> = (0..self.num_vertices()).collect();
        let mut actual_cost: u64 = self.get_route_cost(&actual_route)?;
        let mut best_route: Vec<usize> = (0..self.num_vertices()).collect();
        let mut best_cost: u64 = actual_cost.clone();
        let mut counter: usize = 0;

        let start_time = Instant::now();

        while actual_cost == best_cost {
            for i in 0..self.num_vertices() {
                for j in i + 1..self.num_vertices() {
                    actual_slice = actual_route[i..j].to_vec();
                    actual_slice.reverse();
                    actual_route.splice(i..j, actual_slice);
                    
                    actual_cost = self.get_route_cost(&actual_route)?;

                    if actual_cost < best_cost {
                        best_cost = actual_cost.clone();
                        best_route = actual_route.clone();
                    }
                }
            }

            if log && counter % 10000 == 0{
                println!("Iteration: {} - Route cost: {} - Route found: {}", counter, best_cost, best_route.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" -> "));
            }
            counter += 1;
        }

        let end_time = Instant::now();
        if log {
            println!("Time elapsed: {} - Total iterations: {}", end_time.duration_since(start_time).as_micros(), counter);
        }

        Ok((best_route, best_cost))
    }

    pub fn tsp_3_opt_approx(&self, log: bool) -> Result<(Vec<usize>, u64), &'static str> {
        todo!()
    }

    pub fn tsp_christofides_approx(&self) -> Result<Vec<usize>, &'static str> {
        if !self.is_squared() {
            return Err("Graph is not squared!");
        }

        if !self.is_symmetric() {
            return Err("Graph is not symmetric!");
        }

        let start_time = Instant::now();

        let mut mst: Graph<V> = self.get_mst_prim();
        let mut odd_degree_vertices: Vec<usize> = mst.get_odd_degree_vertices()?;
        let mut odd_graph: Graph<V> = Graph::new();

        for i in 0..odd_degree_vertices.len() {
            for j in 0..odd_degree_vertices.len() {
                odd_graph.insert_edge(odd_degree_vertices[i], odd_degree_vertices[j], self.get_edge_weight(odd_degree_vertices[i], odd_degree_vertices[j])?, false)?;
            }
        }

        let mut matching: Graph<V> = odd_graph.perfect_matching()?;
        mst.union(matching)?;

        // Euleryan cycle
        let mut eulerian_path: Vec<usize> = mst.get_eulerian_path()?;

        let mut end_time = Instant::now();

        if log {
            println!("MST time elapsed: {}", end_time.duration_since(start_time).as_micros());
        }   
    }

    pub fn tsp_nearest_neighbor_greedy(&self) -> Result<Vec<usize>, &'static str> {
        if !self.is_squared() {
            return Err("Graph is not squared!");
        }

        let mut actual_vertex: usize = 0;
        let mut actual_route: Vec<usize> = Vec::new();
        let mut actual_cost: u64;
        let mut min_cost: u64;
        let mut min_cost_vertex: usize;

        // Generating starting point
        actual_route.push(0);

        for i in 1..self.num_vertices() {
            min_cost = self.get_edge_weight(actual_vertex, i)?;
            min_cost_vertex = i;

            for j in 0..self.num_vertices() {
                if actual_route.contains(&j) {
                    continue;
                }

                actual_cost = self.get_edge_weight(actual_vertex, j)?;

                if actual_cost < min_cost {
                    min_cost = actual_cost;
                    min_cost_vertex = j;
                }
            }

            actual_vertex = min_cost_vertex.clone();
            actual_route.push(min_cost_vertex);
        }

        todo!()
    }
}

// https://doc.rust-lang.org/book/ch11-01-writing-tests.html
// https://doc.rust-lang.org/rustc/tests/index.html
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exemple_graph_1() {
        let mut graph: Graph = Graph::new();

        let _ = graph.insert_vertex("A");
        let _ = graph.insert_vertex("B");
        let _ = graph.insert_vertex("C");
        let _ = graph.insert_vertex("D");
        let _ = graph.insert_vertex("E");
        let _ = graph.insert_vertex("F");
        let _ = graph.insert_edge(0, 1, 2, false);
        let _ = graph.insert_edge(0, 2, 1, false);
        let _ = graph.insert_edge(1, 3, 1, false);
        let _ = graph.insert_edge(2, 3, 3, false);
        let _ = graph.insert_edge(2, 4, 4, false);
        let _ = graph.insert_edge(4, 5, 2, false);
        let _ = graph.insert_edge(3, 5, 2, false);

        let path = graph.get_dijkstra_path(0, 5).unwrap();

        for vertex in path.iter() {
            print!("{} ", graph.get_vertex(*vertex).unwrap());
        }
    }

    #[test]
    fn exemple_graph_2() {
        let mut g2 = Graph::new();

        let _ = g2.insert_vertex("A");
        let _ = g2.insert_vertex("B");
        let _ = g2.insert_vertex("C");
        let _ = g2.insert_vertex("D");
        let _ = g2.insert_vertex("E");
        let _ = g2.insert_vertex("F");
        let _ = g2.insert_edge(0, 1, 10, false);
        let _ = g2.insert_edge(0, 3, 5, false);
        let _ = g2.insert_edge(3, 1, 3, false);
        let _ = g2.insert_edge(1, 2, 1, false);
        let _ = g2.insert_edge(3, 2, 8, false);
        let _ = g2.insert_edge(3, 4, 2, false);
        let _ = g2.insert_edge(4, 5, 6, false);
        let _ = g2.insert_edge(2, 4, 4, false);
        let _ = g2.insert_edge(2, 5, 4, false);

        let path = g2.get_dijkstra_path(0, 5).unwrap();
        let min_tree = g2.get_mst_kruskal();

        println!("{:?}", min_tree);

        for vertex in path.iter() {
            print!("{} ", g2.get_vertex(*vertex).unwrap());
        }
    }
}
