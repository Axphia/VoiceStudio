use anyhow::Result;
use burn::backend::{
    wgpu::{Wgpu, WgpuDevice},
    Autodiff, NdArray,
};
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, ElementConversion, Shape, Tensor};
use clap::{Parser, Subcommand};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "voice-burn")]
#[command(about = "Pure Rust Deep Learning Audio & Voice Model Trainer via Burn")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train a Deep Learning Voice Feature & Mood Model in pure Rust
    Train {
        /// Number of training epochs
        #[arg(short, long, default_value_t = 10)]
        epochs: usize,

        /// Batch size
        #[arg(short, long, default_value_t = 16)]
        batch_size: usize,

        /// Learning rate
        #[arg(short, long, default_value_t = 0.001)]
        lr: f64,

        /// Number of input audio features (e.g. 80 Mel bins or 32 spectral features)
        #[arg(long, default_value_t = 32)]
        input_dim: usize,

        /// Number of target voice/mood classes
        #[arg(long, default_value_t = 4)]
        classes: usize,

        /// Use GPU (WGPU) or CPU (NdArray)
        #[arg(long)]
        cpu: bool,
    },

    /// Run inference benchmark with synthetic audio embeddings
    Benchmark {
        #[arg(short, long, default_value_t = 100)]
        iterations: usize,
    },
}

/// A Multi-Layer Perceptron (MLP) Voice Acoustic & Mood Network in Burn
#[derive(Module, Debug)]
pub struct VoiceNet<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    act: Relu,
    dropout: Dropout,
}

impl<B: Backend> VoiceNet<B> {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(input_dim, hidden_dim).init(device),
            fc2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            fc3: LinearConfig::new(hidden_dim, output_dim).init(device),
            act: Relu::new(),
            dropout: DropoutConfig::new(0.1).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        let x = self.dropout.forward(x);
        let x = self.fc2.forward(x);
        let x = self.act.forward(x);
        self.fc3.forward(x)
    }
}

fn run_training_gpu(epochs: usize, batch_size: usize, lr: f64, input_dim: usize, classes: usize) {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    println!("⚡ Using GPU Acceleration via WGPU Backend (DirectX/Vulkan on Windows)!");
    execute_train_loop::<MyAutodiffBackend>(epochs, batch_size, lr, input_dim, classes, &device);
}

fn run_training_cpu(epochs: usize, batch_size: usize, lr: f64, input_dim: usize, classes: usize) {
    type MyBackend = NdArray;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = Default::default();
    println!("🖥️  Using CPU Backend via NdArray (Multi-threaded)!");
    execute_train_loop::<MyAutodiffBackend>(epochs, batch_size, lr, input_dim, classes, &device);
}

fn execute_train_loop<B: burn::tensor::backend::AutodiffBackend>(
    epochs: usize,
    batch_size: usize,
    lr: f64,
    input_dim: usize,
    classes: usize,
    device: &B::Device,
) {
    println!("\n🧠 Initializing VoiceNet in pure Rust (Burn Framework)...");
    let hidden_dim = 64;
    let mut model = VoiceNet::<B>::new(input_dim, hidden_dim, classes, device);
    let mut optim = AdamWConfig::new().with_weight_decay(1e-4).init();

    println!("📊 Architecture: {} -> {} -> {} -> {} (classes)", input_dim, hidden_dim, hidden_dim, classes);
    println!("⚙️  Hyperparams: Epochs={}, BatchSize={}, LearningRate={}\n", epochs, batch_size, lr);

    let t_start = Instant::now();
    let num_batches_per_epoch = 10;

    for epoch in 1..=epochs {
        let t_epoch = Instant::now();
        let mut total_loss = 0.0f32;

        for _ in 0..num_batches_per_epoch {
            // Generate synthetic audio feature batch: (batch_size, input_dim)
            let x: Tensor<B, 2> = Tensor::random([batch_size, input_dim], Distribution::Normal(0.0, 1.0), device);
            // Target one-hot or regression logits
            let y: Tensor<B, 2> = Tensor::random([batch_size, classes], Distribution::Uniform(0.0, 1.0), device);

            // Forward pass
            let preds = model.forward(x);

            // Compute Mean Squared Error (MSE) loss
            let diff = preds - y;
            let loss = (diff.clone() * diff).mean();
            let loss_val: f32 = loss.clone().into_scalar().elem();
            total_loss += loss_val;

            // Backward pass (Autodiff)
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            // Optimizer step
            model = optim.step(lr, model, grads);
        }

        let avg_loss = total_loss / num_batches_per_epoch as f32;
        let epoch_ms = t_epoch.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  [Epoch {:02}/{:02}] Loss: {:.5} | Time: {:.1}ms (Forward + Backward + AdamW)",
            epoch, epochs, avg_loss, epoch_ms
        );
    }

    let elapsed = t_start.elapsed();
    println!("\n🎉 Training finished successfully in {:.2}s!", elapsed.as_secs_f64());
    println!("💾 Model parameters updated in memory using 100% Rust memory safety.\n");
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            epochs,
            batch_size,
            lr,
            input_dim,
            classes,
            cpu,
        } => {
            if cpu {
                run_training_cpu(epochs, batch_size, lr, input_dim, classes);
            } else {
                run_training_gpu(epochs, batch_size, lr, input_dim, classes);
            }
        }
        Commands::Benchmark { iterations } => {
            println!("⚡ Running VoiceNet Inference Benchmark (Pure Rust)...");
            type MyBackend = NdArray;
            let device = Default::default();
            let model = VoiceNet::<MyBackend>::new(32, 64, 4, &device);

            let t0 = Instant::now();
            let input: Tensor<MyBackend, 2> = Tensor::random([1, 32], Distribution::Normal(0.0, 1.0), &device);

            for _ in 0..iterations {
                let _out = model.forward(input.clone());
            }

            let total_time = t0.elapsed();
            let per_iter = (total_time.as_secs_f64() * 1000.0) / iterations as f64;
            println!("Finished {} iterations in {:.2}ms ({:.3}ms per forward pass)", iterations, total_time.as_secs_f64() * 1000.0, per_iter);
        }
    }

    Ok(())
}
