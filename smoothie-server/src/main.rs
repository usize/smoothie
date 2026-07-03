#[cfg(unix)]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use clap::Parser;
use tracing::info;

praxis_filter::register_filters! {
    http "smoothie" => smoothie_filter::SmoothieFilter::from_config,
}

/// Praxis proxy server with Smoothie latency-floor concurrency controller.
#[derive(Parser)]
#[command(name = "smoothie-server")]
struct Cli {
    /// Path to the YAML configuration file.
    #[arg(short = 'c', long = "config")]
    config: Option<String>,
}

fn main() {
    let cli = Cli::parse();
    let explicit = cli.config.or_else(|| std::env::var("PRAXIS_CONFIG").ok());
    let config_path = praxis::resolve_config_path(explicit.as_deref());
    let config =
        praxis::load_config(explicit.as_deref()).unwrap_or_else(|e| praxis::fatal(&e));
    praxis::init_tracing(&config).unwrap_or_else(|e| praxis::fatal(&e));
    info!("starting smoothie-server");
    praxis::run_server_with_registry(config, custom_registry(), config_path)
}
