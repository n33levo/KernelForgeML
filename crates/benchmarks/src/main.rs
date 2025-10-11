//! Benchmark harness executable for KernelForgeML.

use anyhow::Result;
use clap::Parser;
use kernelforge_compiler::cli::{run_cli, Cli};

fn main() -> Result<()> {
    let cli = Cli::parse();
    run_cli(cli)
}
