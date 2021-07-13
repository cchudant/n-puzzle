use anyhow::{anyhow, ensure, Error, Result};
use indexmap::{map::Entry, IndexMap};
use rand::prelude::*;
use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    convert::TryInto,
    fmt,
    hash::Hash,
    io::{stdin, Read},
    iter::Peekable,
    str::FromStr,
    vec,
};
use structopt::StructOpt;

// i took inspiration from
// https://github.com/samueltardieu/pathfinding/blob/main/src/directed/astar.rs
//  (mainly: using indexmap (perf!) & efficient astar impl structure)
//  because my a* implementation was sooo slow (it was using Rc & RefCell and it's very hard to avoid using these
//  unless you have access to indexmap or something like it)

#[derive(Debug, Clone)]
struct SmallestCostHolder {
    estimated_cost: usize,
    cost: usize,
    index: usize,
}

impl PartialEq for SmallestCostHolder {
    fn eq(&self, other: &Self) -> bool {
        self.estimated_cost.eq(&other.estimated_cost) && self.cost.eq(&other.cost)
    }
}

impl Eq for SmallestCostHolder {}

impl PartialOrd for SmallestCostHolder {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SmallestCostHolder {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => self.cost.cmp(&other.cost),
            s => s,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NPuzzleSolver {
    tiles: Board,
    target: Board,
    n: usize,
}

pub struct AStarResult {
    pub states: Option<Vec<Board>>,
    pub open_set_counter: usize,
    pub max_states_in_mem: usize,
}

impl NPuzzleSolver {
    pub fn new(n: usize) -> NPuzzleSolver {
        assert!(n >= 2);
        let mut puzzle = NPuzzleSolver {
            n,
            tiles: Board::new(n),
            target: Board::new(n),
        };
        puzzle.gen_target();

        puzzle
    }

    pub fn from_str(ite: impl Iterator<Item = char>) -> Result<NPuzzleSolver> {
        let mut peekable = ite.peekable();

        fn finish_line(peekable: &mut Peekable<impl Iterator<Item = char>>) -> bool {
            let mut line_finished = false;
            loop {
                match peekable.peek() {
                    Some('#') => {
                        peekable.next();
                        while !matches!(peekable.peek(), Some('\n')) {
                            peekable.next();
                        }
                    }
                    Some('\n') => {
                        line_finished = true;
                        peekable.next();
                    }
                    Some(' ') | Some('\t') | Some('\r') => {
                        peekable.next();
                    }
                    _ => break line_finished,
                }
            }
        };

        fn eat_whitespaces(peekable: &mut Peekable<impl Iterator<Item = char>>) {
            while peekable
                .peek()
                .map_or(false, |c| matches!(c, ' ' | '\t' | '\r'))
            {
                peekable.next();
            }
        };

        fn eat_number(peekable: &mut Peekable<impl Iterator<Item = char>>) -> Option<u32> {
            if !matches!(peekable.peek(), Some('0'..='9')) {
                return None;
            }

            let mut n = 0;
            while matches!(peekable.peek(), Some('0'..='9')) {
                let c = peekable.next().unwrap();
                n = n * 10 + c.to_digit(10).unwrap();
            }
            Some(n)
        };

        finish_line(&mut peekable);
        let n = eat_number(&mut peekable)
            .ok_or_else(|| anyhow!("expected size of puzzle"))?
            .try_into()?;
        ensure!(n > 2, "size of puzzle must be greater than 2");

        let mut puzzle = NPuzzleSolver::new(n);
        for y in 0..n {
            ensure!(finish_line(&mut peekable), "expected newline");
            for x in 0..n {
                eat_whitespaces(&mut peekable);
                let n = eat_number(&mut peekable).ok_or_else(|| anyhow!("expected number"))?;
                puzzle.tiles.0[puzzle.n * y + x] = n.try_into()?;
                eat_whitespaces(&mut peekable);
            }
        }
        finish_line(&mut peekable);
        ensure!(peekable.peek() == None, "expected EOF");

        for num in 0..(n * n) {
            ensure!(
                puzzle.tiles.0.iter().any(|t| *t as usize == num),
                "the number {} is missing from the puzzle",
                num
            );
        }

        Ok(puzzle)
    }

    pub fn random(n: usize, rng: &mut ThreadRng) -> NPuzzleSolver {
        let mut puzzle = NPuzzleSolver::new(n);
        puzzle.tiles.0.copy_from_slice(&puzzle.target.0);

        for _ in 0..128 {
            let empty_tile = puzzle.tiles.0.iter().position(|c| *c == 0).unwrap();
            let moves = Self::possible_moves(empty_tile, n);
            let possible_moves = moves.iter().copied().filter_map(|e| e).collect::<Vec<_>>();
            let mv = possible_moves[rng.gen_range(0..possible_moves.len())];
            puzzle.tiles.0.swap(empty_tile, mv);
        }

        puzzle
    }

    fn gen_target(&mut self) {
        let mut x = 0;
        let mut y = 0;
        let mut direction = 0; // Right, Bottom, Left, Top
        for i in 1..(self.n * self.n) {
            self.target.0[y * self.n + x] = i as u16;

            let change_dir = match direction {
                0 if x == self.n - 1 || self.target.0[(y + 0) * self.n + (x + 1)] != 0 => true,
                1 if y == self.n - 1 || self.target.0[(y + 1) * self.n + (x + 0)] != 0 => true,
                2 if x == 0 || self.target.0[(y + 0) * self.n + (x - 1)] != 0 => true,
                3 if y == 0 || self.target.0[(y - 1) * self.n + (x + 0)] != 0 => true,
                _ => false,
            };

            if change_dir {
                direction = (direction + 1) % 4;
            }

            match direction {
                0 => x += 1,
                1 => y += 1,
                2 => x -= 1,
                3 => y -= 1,
                _ => unreachable!(),
            }
        }
    }

    pub fn get_n(&self) -> usize {
        self.n
    }

    pub fn index_to_xy(index: usize, n: usize) -> (usize, usize) {
        (index % n, index / n)
    }

    fn possible_moves(empty_tile: usize, n: usize) -> [Option<usize>; 4] {
        [
            {
                if empty_tile % n == n - 1 {
                    None
                } else {
                    Some(empty_tile + 1)
                }
            },
            {
                if empty_tile % n == 0 {
                    None
                } else {
                    Some(empty_tile - 1)
                }
            },
            {
                if empty_tile / n == n - 1 {
                    None
                } else {
                    Some(empty_tile + n)
                }
            },
            {
                if empty_tile / n == 0 {
                    None
                } else {
                    Some(empty_tile - n)
                }
            },
        ]
    }

    fn astar_successors(board: &Board, n: usize) -> Vec<Board> {
        let empty_tile = board.0.iter().position(|c| *c == 0).unwrap();
        let arr = Self::possible_moves(empty_tile, n);

        arr.iter()
            .copied()
            .filter_map(|e| e)
            .map(|mv| {
                let mut board = board.clone();
                board.0.swap(empty_tile, mv);
                board
            })
            .collect()
    }

    pub fn perform_astar(&mut self, h: Heuristics) -> AStarResult {
        let mut to_see = BinaryHeap::new();
        let mut open_set_counter = 1; // initial state: 1 node in open set
        to_see.push(SmallestCostHolder {
            cost: 0,
            index: 0,
            estimated_cost: 0,
        });

        let mut parents: IndexMap<Board, (usize, usize)> = IndexMap::default();
        parents.insert(self.tiles.clone(), (usize::max_value(), 0));

        while let Some(SmallestCostHolder { cost, index, .. }) = to_see.pop() {
            let successors = {
                let (node, &(_, c)) = parents.get_index(index).unwrap();
                if &self.target == node {
                    // reverse path

                    let mut vec = Vec::new();
                    let mut index = index;

                    while let Some((node, &(ind, _))) = parents.get_index(index) {
                        index = ind;
                        vec.push(node.clone());
                    }
                    return AStarResult {
                        states: Some(vec),
                        open_set_counter,
                        max_states_in_mem: parents.len(),
                    };
                }
                if cost > c {
                    continue;
                }

                Self::astar_successors(node, self.n)
            };

            for successor in successors {
                let new_cost = cost + 1; // move_cost = 1
                let heu;
                let succ_index;

                match parents.entry(successor) {
                    Entry::Vacant(e) => {
                        heu = h.run_heuristic(e.key(), &self.target, self.n);
                        succ_index = e.index();
                        e.insert((index, new_cost));
                    }
                    Entry::Occupied(mut e) => {
                        if e.get().1 > new_cost {
                            heu = h.run_heuristic(e.key(), &self.target, self.n);
                            succ_index = e.index();
                            e.insert((index, new_cost));
                        } else {
                            continue;
                        }
                    }
                }

                to_see.push(SmallestCostHolder {
                    estimated_cost: new_cost + heu,
                    cost: new_cost,
                    index: succ_index,
                });
                open_set_counter += 1;
            }
        }

        AStarResult {
            states: None,
            open_set_counter,
            max_states_in_mem: parents.len(),
        }
    }
}

impl fmt::Display for NPuzzleSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tiles)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Board(pub Vec<u16>);
impl Board {
    pub fn new(n: usize) -> Board {
        Board(vec![0; n * n])
    }
}
impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = (self.0.len() as f64).sqrt() as usize;
        let n_width = (*self.0.iter().max().unwrap() as f64).log10() as usize + 1;

        let fmt_tile = |t: u16, f: &mut fmt::Formatter<'_>| match t {
            0 => write!(f, "{:^width$}", "", width = n_width),
            n => write!(f, "{:^width$}", n, width = n_width),
        };

        for y in 0..n {
            if y == 0 {
                write!(f, "[")?;
            } else {
                write!(f, " ")?;
            }
            write!(f, "[ ")?;
            for x in 0..n {
                fmt_tile(self.0[y * n + x], f)?;
                write!(f, " ")?;
            }
            write!(f, "]")?;
            if y == n - 1 {
                write!(f, "]\n")?;
            } else {
                write!(f, ",\n")?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, StructOpt)]
pub enum Heuristics {
    ManhattanDistance,
    MisplacedTiles,
    EuclideanDistanceSquared,
}

impl FromStr for Heuristics {
    type Err = Error;
    fn from_str(day: &str) -> Result<Self, Self::Err> {
        match day {
            "manhattan" => Ok(Heuristics::ManhattanDistance),
            "misplaced" => Ok(Heuristics::ManhattanDistance),
            "euclidean" => Ok(Heuristics::ManhattanDistance),
            _ => Err(anyhow!("Could not parse heuristic")),
        }
    }
}

impl fmt::Display for Heuristics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Heuristics::ManhattanDistance => "manhattan distance",
                Heuristics::MisplacedTiles => "number of misplaced tiles",
                Heuristics::EuclideanDistanceSquared => "euclidean distance squared",
            }
        )
    }
}

impl Heuristic for Heuristics {
    fn run_heuristic(&self, puzzle: &Board, target: &Board, n: usize) -> usize {
        match *self {
            Heuristics::ManhattanDistance => {
                ManhattanDistanceHeuristic.run_heuristic(puzzle, target, n)
            }
            Heuristics::MisplacedTiles => MisplacedTilesHeuristic.run_heuristic(puzzle, target, n),
            Heuristics::EuclideanDistanceSquared => {
                EuclideanDistanceSquaredHeuristic.run_heuristic(puzzle, target, n)
            }
        }
    }
}

pub trait Heuristic {
    fn run_heuristic(&self, puzzle: &Board, target: &Board, n: usize) -> usize;
}

struct MisplacedTilesHeuristic;
impl Heuristic for MisplacedTilesHeuristic {
    fn run_heuristic(&self, puzzle: &Board, target: &Board, _n: usize) -> usize {
        puzzle
            .0
            .iter()
            .zip(&target.0)
            .filter(|(t, tg)| t != tg)
            .count()
    }
}

struct ManhattanDistanceHeuristic;
impl Heuristic for ManhattanDistanceHeuristic {
    fn run_heuristic(&self, puzzle: &Board, target: &Board, n: usize) -> usize {
        let mut res = 0;
        for (index, el) in puzzle.0.iter().enumerate() {
            let index_tg = target.0.iter().position(|t| t == el).unwrap();
            let (x, y) = NPuzzleSolver::index_to_xy(index, n);
            let (x_tg, y_tg) = NPuzzleSolver::index_to_xy(index_tg, n);

            res +=
                ((x as isize - x_tg as isize).abs() + (y as isize - y_tg as isize).abs()) as usize;
        }
        res
    }
}

struct EuclideanDistanceSquaredHeuristic;
impl Heuristic for EuclideanDistanceSquaredHeuristic {
    fn run_heuristic(&self, puzzle: &Board, target: &Board, n: usize) -> usize {
        let mut res = 0;
        for (index, el) in puzzle.0.iter().enumerate() {
            let index_tg = target.0.iter().position(|t| t == el).unwrap();
            let (x, y) = NPuzzleSolver::index_to_xy(index, n);
            let (x_tg, y_tg) = NPuzzleSolver::index_to_xy(index_tg, n);

            res += ((x as isize - x_tg as isize).pow(2) + (y as isize - y_tg as isize).pow(2))
                as usize;
        }
        res
    }
}

#[derive(Debug, Clone, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
pub struct Args {
    /// The heuristic to use (possible values: manhattan, misplaced, euclidean)
    #[structopt(short, long, default_value = "manhattan")]
    pub heuristic: Heuristics,

    /// The size of the puzzle to generate.
    ///
    /// If set, the program will not read the puzzle from stdin, but will create
    /// a random puzzle instead.
    #[structopt(short, long)]
    pub random_size: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::from_args();

    let mut puzzle = if let Some(n) = args.random_size {
        NPuzzleSolver::random(n, &mut rand::thread_rng())
    } else {
    println!("Reading puzzle from stdin...");
    let mut buf = String::new();
        stdin().read_to_string(&mut buf)?;
        NPuzzleSolver::from_str(buf.chars())?
    };

    println!("-- Initial state --\n{}", puzzle.tiles);

    println!("Solving...\n");

    let heuristic = args.heuristic;
    let result = puzzle.perform_astar(heuristic);

    if let Some(states) = result.states {
        println!("-- Solving sequence --");
        for (ind, state) in states.iter().rev().skip(1).enumerate() {
            println!("move #{:03}:\n{}", ind + 1, state);
        }

        println!("-- Summary --");
        println!("heuristic: {}", heuristic);
        println!("solvable: yes");
        println!("number of moves required: {}", states.len() - 1);
        println!(
            "total number of states ever selected in the open set: {}",
            result.open_set_counter
        );
        println!(
            "maximum number of states ever represented in memory at the same time: {}",
            result.max_states_in_mem
        );
    } else {
        println!("-- Summary --");
        println!("heuristic: {}", heuristic);
        println!("solvable: no");
        println!("number of moves required: N/A");
        println!(
            "total number of states ever selected in the open set: {}",
            result.open_set_counter
        );
        println!(
            "maximum number of states ever represented in memory at the same time: {}",
            result.max_states_in_mem
        );
    }

    Ok(())
}
