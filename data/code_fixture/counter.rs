// SPDX-License-Identifier: MIT
use std::collections::BTreeMap;

#[derive(Default)]
pub struct Counter {
    values: BTreeMap<u8, u32>,
}

impl Counter {
    pub fn observe(&mut self, byte: u8) {
        *self.values.entry(byte).or_insert(0) += 1;
    }

    pub fn total(&self) -> u32 {
        self.values.values().sum()
    }

    pub fn most_common(&self) -> Option<(u8, u32)> {
        self.values
            .iter()
            .max_by_key(|(byte, count)| (**count, **byte))
            .map(|(byte, count)| (*byte, *count))
    }
}

pub fn count_bytes(data: &[u8]) -> Counter {
    let mut counter = Counter::default();
    for &byte in data {
        counter.observe(byte);
    }
    counter
}
