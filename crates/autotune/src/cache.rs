//! Result caching for autotuning.

use anyhow::Result;
use kernelforge_kernels::config::{KernelProfile, MatmulProblem};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AutotuneCache {
    matmul: HashMap<String, KernelProfile>,
}

impl AutotuneCache {
    pub fn new() -> Self {
        Self {
            matmul: HashMap::new(),
        }
    }

    pub fn get_matmul(&self, problem: &MatmulProblem) -> Option<&KernelProfile> {
        self.matmul.get(&problem_key(problem))
    }

    pub fn insert_matmul(&mut self, profile: KernelProfile) {
        let key = problem_key(&profile.problem);
        self.matmul.insert(key, profile);
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let data = fs::read(path)?;
        let cache = serde_json::from_slice(&data)?;
        Ok(cache)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let blob = serde_json::to_vec_pretty(self)?;
        fs::write(path, blob)?;
        Ok(())
    }
}

fn problem_key(problem: &MatmulProblem) -> String {
    serde_json::to_string(problem).unwrap_or_else(|_| "invalid-problem".to_string())
}
