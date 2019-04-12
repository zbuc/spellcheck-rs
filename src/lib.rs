
//use num_traits::pow::Pow;
// use assert_approx_eq::assert_approx_eq;

use std::f64::consts::E;
use std::hash::Hash;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use fasthash::murmur3;

fn hash<T: Hash +std::convert::AsRef<[u8]>>(t: &T, seed: u32) -> u128 {
    murmur3::hash128_with_seed(t, seed)
}

/// This module is a python module implemented in Rust
#[pymodule]
fn bloomfilter(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBloomFilter>()?;

    Ok(())
}

#[pyclass]
pub struct PyBloomFilter {
    bf: BloomFilter,
}

#[pymethods]
impl PyBloomFilter {
    #[new]
    pub fn new(obj: &PyRawObject, max_capacity: usize, target_error_rate: f64) {
        obj.init(PyBloomFilter {
            bf: BloomFilter::new(max_capacity, target_error_rate),
        });
    }

    /// Adds an element to an existing BloomFilter from Python
    fn add(&mut self, item: String) -> PyResult<bool> {
        self.bf.add(item.as_bytes());
        Ok(true)
    }

    /// Checks for an element in an existing BloomFilter from Python
    fn check_existence(&self, item: String) -> PyResult<bool> {
        Ok(self.bf.check_existence(item.as_bytes()))
    }

    /// Returns the error rate of the BloomFilter as configured from Python
    fn error_rate(&self) -> PyResult<f64> {
        Ok(self.bf.error_rate())
    }
}

// https://codeburst.io/lets-implement-a-bloom-filter-in-go-b2da8a4b849f
pub struct BloomFilter {
    pub bitset: Vec<bool>, // probably want this on the stack -- also the size of bool is 1 byte, we could use a bitfield instead
    pub k: usize, // the number of hash values
    pub n: usize, // the number of elements in the filter
    pub m: usize, // size of the bloom filter bitset
    // we *could* support arbitrary hash functions, doing something like this
    // pub hashfuncs: Vec<&'call(Fn(str) -> usize + 'call)>,
    // but for simplicity and efficiency, we'll *always* use murmur3
    // and we only need to store the seeds
    // https://play.rust-lang.org/?code=trait%20Barable%20%7B%0A%20%20%20%20fn%20new()%20-%3E%20Self%3B%0A%20%20%20%20%2F%2F%20etc.%0A%7D%0A%0Astruct%20Bar%3B%0A%0Aimpl%20Barable%20for%20Bar%20%7B%0A%20%20%20%20fn%20new()%20-%3E%20Self%20%7B%0A%20%20%20%20%20%20%20%20Bar%0A%20%20%20%20%7D%0A%7D%0A%0Astruct%20Foo%3CB%20%3ABarable%3E%20%7B%0A%20%20%20%20bar%3A%20B%2C%0A%20%20%20%20callback%3A%20fn(%26B)%2C%0A%7D%0A%0Aimpl%3CB%3A%20Barable%3E%20Foo%3CB%3E%20%7B%0A%20%20%20%20fn%20new(callback%3A%20fn(%26B))%20-%3E%20Self%20%7B%0A%20%20%20%20%20%20%20%20Foo%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20bar%3A%20B%3A%3Anew()%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20callback%3A%20callback%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%7D%0A%7D%0A%0Afn%20example_callback(bar%3A%20%26Bar)%20%7B%0A%7D%0A%0Afn%20main()%20%7B%0A%20%20%20%20let%20foo%20%3D%20Foo%3A%3Anew(example_callback)%3B%0A%7D&version=stable&backtrace=0
    pub murmur3_seeds: Vec<usize>,
}

// All of the Vecs here are heap-allocated. We might speed things up by
// putting them on the stack.
impl BloomFilter {
    pub fn new(max_capacity: usize, target_error_rate: f64) -> Self {
        let bitset_size = BloomFilter::optimal_bitset_size(max_capacity, target_error_rate);
        let num_hashfuncs = BloomFilter::hashfunctions_needed(bitset_size, max_capacity);

        BloomFilter {
            bitset: vec![false; bitset_size],
            k: num_hashfuncs,
            m: bitset_size,
            n: 0,
            murmur3_seeds: (0..num_hashfuncs).step_by(1).collect(),
        }
    }

    pub fn add(&mut self, item: &[u8]) {
        let hashes: Vec<u128> = self.get_hashes(item);

        for i in 0..self.k {
            let position = hashes[i] % self.m as u128;
            self.bitset[position as usize] = true
        }

        self.n = self.n + 1;
    }

    pub fn get_hashes(&self, item: &[u8]) -> Vec<u128> {
        let mut results: Vec<u128> = Vec::new();

        for seed in &self.murmur3_seeds {
            results.push(hash(&item, *seed as u32));
        }

        results
    }

    // If all positions are set, the item possibly exists.
    // If any are not set, the item definitely does not exist.
    pub fn check_existence(&self, item: &[u8]) -> bool {
        let hashes = self.get_hashes(item);

        let mut exists = true;

        for i in 0..self.k {
            let position = hashes[i] % self.m as u128;

            if !self.bitset[position as usize] {
                exists = false;
                break;
            }
        }

        exists
    }

    pub fn error_rate(&self) -> f64 {
        error_rate(self.m as i64, self.k as i64, self.n as i64)
    }
    
    pub fn optimal_bitset_size(capacity: usize, false_positive_rate: f64) -> usize {
        (-1.0 * (false_positive_rate.log(E) * capacity as f64) / (2.0_f64.log(E).powf(2.0))).ceil() as usize
    }

    pub fn hashfunctions_needed(bitset_size: usize, capacity: usize) -> usize {
        ((bitset_size as f64 / capacity as f64) * 2.0_f64.log(E)).ceil() as usize
    }
}

fn error_rate(filter_size: i64, num_hashfunctions: i64, num_elements: i64) -> f64 {
    let filter_size = filter_size as f64;
    let num_hashfunctions = num_hashfunctions as f64;
    let num_elements = num_elements as f64;

    (1.0 - E.powf((-1.0 * num_hashfunctions * num_elements) / filter_size)).powf(num_hashfunctions)
}

#[cfg(test)]
mod tests {
    use crate::error_rate;
    use crate::hash;
    use crate::BloomFilter;
    use fasthash::murmur3;

    use pyo3::prelude::*;
    use pyo3::wrap_pyfunction;

    macro_rules! assert_delta {
        ($x:expr, $y:expr, $d:expr) => {
            if !(($x - $y).abs() < $d) { panic!("{} >= {}", ($x - $y).abs(), $d); }
        }
    }

    #[test]
    fn test_error_rate() {
        // https://www.semantics3.com/blog/use-the-bloom-filter-luke-b59fd0839fc4/
        // https://hackernoon.com/probabilistic-data-structures-bloom-filter-5374112a7832
        assert_eq!(error_rate(100_000, 1, 10_000_000), 1.0);
        assert_eq!(error_rate(100_000, 5, 10_000_000), 1.0);
        assert_delta!(error_rate(100_000_000, 5, 10_000_000), 0.01, 0.0006);
    }

    #[test]
    fn test_hashing() {
        let h = murmur3::hash128_with_seed(b"hello world\xff", 1);

        assert_eq!(h as u128, hash(b"hello world\xff", 1));

        let h = murmur3::hash128_with_seed(b"hello world\xff", 2);

        assert_ne!(h as u128, hash(b"hello world\xff", 1));
    }

    #[test]
    fn test_bloomfilter() {
        let mut bf = BloomFilter::new(1024, 0.01);
        bf.add("hello".as_bytes());
        bf.add("world".as_bytes());
        bf.add("sir".as_bytes());
        bf.add("madam".as_bytes());
        bf.add("io".as_bytes());

        assert!(bf.check_existence("hello".as_bytes()));
        assert!(bf.check_existence("world".as_bytes()));
        assert!(!bf.check_existence("nonexistent".as_bytes()));
    }

    #[test]
    fn test_size_optimization() {
        let p = 0.01;
        let n = 10_000_000;
        let expected_k = 7;
        let expected_m = 95_850_584;

        let m = BloomFilter::optimal_bitset_size(n, p);
        assert_eq!(m, expected_m);
        let k = BloomFilter::hashfunctions_needed(m, n);
        assert_eq!(k, expected_k);
    }
}
