#![recursion_limit = "256"]

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::{Autodiff, NdArray};
use burn::nn::PaddingConfig2d;
use burn::{
    config::Config,
    module::Module,
    nn,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::{Int, Tensor, TensorData, activation::relu, backend::Backend},
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::WavReader;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

type TrainBackend = Autodiff<NdArray>;
type InferenceBackend = NdArray;
// type TrainBackend = Autodiff<Wgpu>;
// type InferenceBackend = Wgpu;

// Audio constants
const SAMPLE_RATE: u32 = 16000;
const FRAME_SIZE: usize = 512;
const HOP_SIZE: usize = 160;

// Model constants - ADJUSTED FOR 3-SECOND CLIPS
const MAX_WORDS_PER_CLIP: usize = 15; // 3 seconds = ~5-10 words typically
const WORD_VOCAB_SIZE: usize = 500; // Start smaller for faster learning
const HIDDEN_DIM: usize = 512; // Increase model capacity
const N_MELS: usize = 80; // More frequency bins
const LSTM_LAYERS: usize = 2; // Stack LSTM layers

// Training constants - IMPROVED
const TRAIN_SPLIT: f32 = 0.8;
const INITIAL_LR: f64 = 0.001;
const MIN_LR: f64 = 0.00001;
const LR_DECAY_RATE: f64 = 0.95; // Decay LR each epoch
const EPOCHS: usize = 100;
const MAX_AUDIO_LENGTH: f32 = 3.0;
const MIN_WORD_COUNT: usize = 2; // Lower threshold
const BATCH_SIZE: usize = 16; // Larger batches
const DROPOUT_RATE: f64 = 0.5; // More regularization

const LEARNING_RATE: f64 = 0.001;
// Special tokens
const PAD_TOKEN: usize = 0;
const UNK_TOKEN: usize = 1;
const SOS_TOKEN: usize = 2;
const EOS_TOKEN: usize = 3;

#[derive(Config)]
pub struct WordWhisperConfig {
    vocab_size: usize,
    feature_dim: usize,
    hidden_dim: usize,
    max_words: usize,
}

#[derive(Module, Debug)]
pub struct WordWhisper<B: Backend> {
    // Enhanced audio encoder
    conv1: nn::conv::Conv1d<B>,
    bn1: nn::BatchNorm<B, 1>,
    conv2: nn::conv::Conv1d<B>,
    bn2: nn::BatchNorm<B, 1>,
    conv3: nn::conv::Conv1d<B>,
    bn3: nn::BatchNorm<B, 1>,
    pool: nn::pool::AdaptiveAvgPool1d,

    // Stacked LSTM layers
    lstm1: nn::lstm::Lstm<B>,
    lstm2: nn::lstm::Lstm<B>,

    // Word prediction with intermediate layer
    pre_proj: nn::Linear<B>,
    word_proj: nn::Linear<B>,
    dropout: nn::Dropout,
}

impl<B: Backend> WordWhisper<B> {
    pub fn new(config: &WordWhisperConfig, device: &B::Device) -> Self {
        // Larger kernels for better temporal modeling
        let conv1 = nn::conv::Conv1dConfig::new(config.feature_dim, 128, 5)
            .with_padding(nn::PaddingConfig1d::Same)
            .init(device);
        let bn1 = nn::BatchNormConfig::new(128).init(device);

        let conv2 = nn::conv::Conv1dConfig::new(128, 256, 5)
            .with_padding(nn::PaddingConfig1d::Same)
            .init(device);
        let bn2 = nn::BatchNormConfig::new(256).init(device);

        let conv3 = nn::conv::Conv1dConfig::new(256, config.hidden_dim, 5)
            .with_padding(nn::PaddingConfig1d::Same)
            .init(device);
        let bn3 = nn::BatchNormConfig::new(config.hidden_dim).init(device);

        let pool = nn::pool::AdaptiveAvgPool1dConfig::new(config.max_words).init();

        // Stacked LSTMs
        let lstm1 =
            nn::lstm::LstmConfig::new(config.hidden_dim, config.hidden_dim, false).init(device);
        let lstm2 =
            nn::lstm::LstmConfig::new(config.hidden_dim, config.hidden_dim, false).init(device);

        // Two-layer prediction head
        let pre_proj = nn::LinearConfig::new(config.hidden_dim, config.hidden_dim / 2).init(device);
        let word_proj =
            nn::LinearConfig::new(config.hidden_dim / 2, config.vocab_size).init(device);
        let dropout = nn::DropoutConfig::new(DROPOUT_RATE).init();

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            pool,
            lstm1,
            lstm2,
            pre_proj,
            word_proj,
            dropout,
        }
    }

    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, mels] = features.dims();

        // Enhanced conv processing with batch norm
        let x = features.swap_dims(1, 2);
        let x = relu(self.bn1.forward(self.conv1.forward(x)));
        let x = self.dropout.forward(x);
        let x = relu(self.bn2.forward(self.conv2.forward(x)));
        let x = self.dropout.forward(x);
        let x = relu(self.bn3.forward(self.conv3.forward(x)));

        // Pool and reshape
        let x = self.pool.forward(x);
        let x = x.swap_dims(1, 2);

        // Stacked LSTM processing
        let x_t = x.swap_dims(0, 1);
        let (x, _) = self.lstm1.forward(x_t, None);
        let x = self.dropout.forward(x);
        let (output, _) = self.lstm2.forward(x, None);
        let output = output.swap_dims(0, 1);

        // Two-layer prediction head
        let output = self.dropout.forward(output);
        let output = relu(self.pre_proj.forward(output));
        let output = self.dropout.forward(output);
        self.word_proj.forward(output)
    }
}

// Build vocabulary from dataset
fn build_vocabulary(dataset_path: &str) -> (Vec<String>, HashMap<String, usize>) {
    println!("📚 Building vocabulary...");

    let mut word_counts: HashMap<String, usize> = HashMap::new();

    // Count words in all transcripts
    for entry in fs::read_dir(dataset_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("txt") {
            if let Ok(text) = fs::read_to_string(&path) {
                for word in text.to_lowercase().split_whitespace() {
                    // Clean word (remove punctuation)
                    let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
                    if !clean_word.is_empty() {
                        *word_counts.entry(clean_word.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Sort by frequency and take top words
    let mut words: Vec<(String, usize)> = word_counts.into_iter().collect();
    words.sort_by(|a, b| b.1.cmp(&a.1));

    // Create vocabulary with special tokens
    let mut vocab = vec![
        "<pad>".to_string(),
        "<unk>".to_string(),
        "<sos>".to_string(),
        "<eos>".to_string(),
    ];

    // Add top frequent words
    for (word, count) in words.iter().take(WORD_VOCAB_SIZE - 4) {
        vocab.push(word.clone());
        if vocab.len() <= 20 {
            println!("  Word {}: '{}' (count: {})", vocab.len() - 4, word, count);
        }
    }

    // Create word to index mapping
    let word_to_idx: HashMap<String, usize> = vocab
        .iter()
        .enumerate()
        .map(|(idx, word)| (word.clone(), idx))
        .collect();

    println!("✅ Vocabulary size: {}", vocab.len());

    (vocab, word_to_idx)
}

// Load and preprocess audio
fn load_audio(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = WavReader::open(path).map_err(|e| format!("Failed to open WAV: {}", e))?;

    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != SAMPLE_RATE {
        return Err(format!(
            "Expected mono {}Hz, got {} channels at {}Hz",
            SAMPLE_RATE, spec.channels, spec.sample_rate
        ));
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    Ok(samples)
}

// Compute mel spectrogram
fn compute_mel_spectrogram(audio: &[f32]) -> Vec<Vec<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FRAME_SIZE);
    let mut mel_frames = Vec::new();

    // Create simple mel filterbank
    let mel_filters = create_mel_filterbank();

    for i in (0..audio.len().saturating_sub(FRAME_SIZE)).step_by(HOP_SIZE) {
        let frame = &audio[i..i + FRAME_SIZE];

        // Apply window
        let windowed: Vec<Complex<f32>> = frame
            .iter()
            .enumerate()
            .map(|(j, &sample)| {
                let window = 0.5
                    - 0.5 * (2.0 * std::f32::consts::PI * j as f32 / (FRAME_SIZE - 1) as f32).cos();
                Complex::new(sample * window, 0.0)
            })
            .collect();

        let mut fft_buffer = windowed;
        fft.process(&mut fft_buffer);

        // Convert to power and apply mel filters
        let power: Vec<f32> = fft_buffer[..FRAME_SIZE / 2]
            .iter()
            .map(|c| c.norm_sqr().max(1e-10))
            .collect();

        let mut mel_frame = vec![0.0; N_MELS];
        for (i, filter) in mel_filters.iter().enumerate() {
            mel_frame[i] = power
                .iter()
                .zip(filter.iter())
                .map(|(p, f)| p * f)
                .sum::<f32>()
                .log10();
        }

        mel_frames.push(mel_frame);
    }

    mel_frames
}

// Simple mel filterbank
fn create_mel_filterbank() -> Vec<Vec<f32>> {
    let mut filterbank = vec![vec![0.0; FRAME_SIZE / 2]; N_MELS];

    for i in 0..N_MELS {
        let center = (i + 1) * (FRAME_SIZE / 2) / (N_MELS + 1);
        let width = FRAME_SIZE / (2 * N_MELS);

        for j in 0..FRAME_SIZE / 2 {
            if j >= center.saturating_sub(width) && j <= center + width {
                let distance = (j as i32 - center as i32).abs() as f32;
                filterbank[i][j] = 1.0 - (distance / width as f32);
            }
        }
    }

    filterbank
}

// Convert text to word indices
fn text_to_word_indices(text: &str, word_to_idx: &HashMap<String, usize>) -> Vec<usize> {
    let mut indices = vec![SOS_TOKEN];

    for word in text.to_lowercase().split_whitespace() {
        let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
        if !clean_word.is_empty() {
            let idx = word_to_idx.get(clean_word).unwrap_or(&UNK_TOKEN);
            indices.push(*idx);
        }
    }

    indices.push(EOS_TOKEN);
    indices
}

// Decode predictions back to text
fn decode_predictions(predictions: Tensor<InferenceBackend, 3>, vocab: &[String]) -> String {
    let pred_data = predictions.argmax(2).into_data();
    let indices = pred_data.as_slice::<i64>().unwrap();

    let mut words = Vec::new();
    for &idx in indices {
        dbg!(idx);
        if idx as usize >= vocab.len() {
            continue;
        }

        let word = &vocab[idx as usize];

        // Skip special tokens and consecutive duplicates
        if word != "<pad>" && word != "<sos>" && word != "<eos>" {
            if words.is_empty() || words.last() != Some(word) {
                words.push(word.clone());
            }
        }
    }

    words.join(" ")
}

// Load dataset
fn load_dataset(
    dataset_path: &str,
    word_to_idx: &HashMap<String, usize>,
) -> Vec<(Vec<Vec<f32>>, Vec<usize>)> {
    println!("\n📁 Loading dataset...");
    let mut data = Vec::new();
    let mut skipped = 0;

    for entry in fs::read_dir(dataset_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("wav") {
            let txt_path = path.with_extension("txt");

            if txt_path.exists() {
                if let Ok(audio) = load_audio(&path) {
                    if let Ok(text) = fs::read_to_string(&txt_path) {
                        // Check audio length
                        let duration = audio.len() as f32 / SAMPLE_RATE as f32;
                        if duration > MAX_AUDIO_LENGTH {
                            skipped += 1;
                            continue;
                        }

                        let word_indices = text_to_word_indices(&text, word_to_idx);

                        // Skip if too few words
                        if word_indices.len() < MIN_WORD_COUNT + 2 {
                            // +2 for SOS/EOS
                            skipped += 1;
                            continue;
                        }

                        let mel_frames = compute_mel_spectrogram(&audio);

                        if mel_frames.len() > 10 {
                            // Clone before moving
                            let word_count = word_indices.len() - 2;
                            let frame_count = mel_frames.len();

                            data.push((mel_frames, word_indices));

                            if data.len() <= 5 {
                                println!(
                                    "  Sample {}: {} words, {} frames",
                                    data.len(),
                                    word_count,
                                    frame_count
                                );
                            }
                        }
                    }
                }
            }
        }

        if data.len() % 100 == 0 && data.len() > 0 {
            print!("\r  Loaded {} samples...", data.len());
            std::io::stdout().flush().unwrap();
        }
    }

    println!(
        "\n✅ Loaded {} samples (skipped {} too long/short)",
        data.len(),
        skipped
    );
    data
}

// Custom loss function that ignores padding
fn compute_loss_with_mask<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    pad_token: i64,
    device: &B::Device,
) -> Tensor<B, 1> {
    let [batch, seq_len, vocab_size] = logits.dims();

    // Reshape for cross entropy
    let logits_flat = logits.reshape([batch * seq_len, vocab_size]);
    let targets_flat = targets.reshape([batch * seq_len]);

    // Create mask for non-padding tokens
    let mask = targets_flat.clone().not_equal_elem(pad_token).float();

    // Compute cross entropy loss
    let loss = nn::loss::CrossEntropyLossConfig::new()
        .init(device)
        .forward(logits_flat, targets_flat);

    // Apply mask and compute mean
    let masked_loss = loss * mask.clone();
    let sum_loss = masked_loss.sum();
    let count = mask.sum().add_scalar(1e-6); // Avoid division by zero

    sum_loss.div(count).reshape([1])
}

// Training function
fn train_model(dataset_path: &str) {
    let device = Default::default();
    // burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
    //     &device,
    //     Default::default(),
    // );

    // Build vocabulary
    let (vocab, word_to_idx) = build_vocabulary(dataset_path);

    // Save vocabulary for later use
    let vocab_json = serde_json::to_string(&vocab).unwrap();
    fs::write("vocabulary.json", vocab_json).expect("Failed to save vocabulary");

    let config = WordWhisperConfig {
        vocab_size: vocab.len(),
        feature_dim: N_MELS,
        hidden_dim: HIDDEN_DIM,
        max_words: MAX_WORDS_PER_CLIP,
    };

    let mut model = WordWhisper::<TrainBackend>::new(&config, &device);
    let mut optimizer = AdamConfig::new().init();

    // Load dataset
    let mut all_data = load_dataset(dataset_path, &word_to_idx);

    if all_data.is_empty() {
        panic!("No data loaded!");
    }

    // Shuffle and split
    let mut rng = thread_rng();
    all_data.shuffle(&mut rng);

    let split_idx = (all_data.len() as f32 * TRAIN_SPLIT) as usize;
    let (train_data, val_data) = all_data.split_at(split_idx);

    println!("\n🎯 Starting training:");
    println!("   Train samples: {}", train_data.len());
    println!("   Val samples: {}", val_data.len());
    println!("   Vocabulary size: {}", vocab.len());
    println!("   Learning rate: {}", LEARNING_RATE);

    let mut best_val_loss = f32::INFINITY;
    let mut current_lr = INITIAL_LR;

    // Training loop
    for epoch in 0..EPOCHS {
        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0;
        let mut epoch_total = 0;
        let start_time = std::time::Instant::now();

        // Shuffle training data
        let mut train_batch: Vec<_> = train_data.to_vec();
        train_batch.shuffle(&mut rng);

        for (mel_frames, word_indices) in train_batch.iter() {
            // Prepare features
            let features_flat: Vec<f32> = mel_frames.iter().flatten().copied().collect();
            let features = Tensor::<TrainBackend, 3>::from_data(
                TensorData::new(features_flat, [1, mel_frames.len(), N_MELS]),
                &device,
            );

            // Prepare targets (pad or truncate to MAX_WORDS_PER_CLIP)
            let mut padded_targets = vec![PAD_TOKEN; MAX_WORDS_PER_CLIP];
            for (i, &idx) in word_indices.iter().enumerate().take(MAX_WORDS_PER_CLIP) {
                padded_targets[i] = idx;
            }

            let targets = Tensor::<TrainBackend, 2, Int>::from_data(
                TensorData::new(
                    padded_targets.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                    [1, MAX_WORDS_PER_CLIP],
                ),
                &device,
            );

            // Forward pass
            let logits = model.forward(features);

            // Calculate accuracy
            let predictions = logits.clone().argmax(2);
            let pred_data = predictions.into_data();
            let pred_indices = pred_data.as_slice::<i64>().unwrap();

            for i in 0..word_indices.len().min(MAX_WORDS_PER_CLIP) {
                if pred_indices[i] == padded_targets[i] as i64 {
                    epoch_correct += 1;
                }
                epoch_total += 1;
            }

            // Compute loss with padding mask
            let loss = compute_loss_with_mask(logits, targets, PAD_TOKEN as i64, &device);

            let loss_value = loss.clone().into_scalar();
            epoch_loss += loss_value;

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(LEARNING_RATE, model, grads);
            // // Decay learning rate
            // current_lr = (current_lr * LR_DECAY_RATE).max(MIN_LR);

            // // Update optimizer with new learning rate
            // model = optimizer.step(current_lr, model, grads);

            // // Add debugging every few epochs
            // if epoch % 10 == 0 && epoch > 0 {
            //     println!("\n📊 Debug - Top 5 predicted words:");
            //     // Sample one batch and show predictions
            //     if let Some((mel_frames, word_indices)) = train_batch.first() {
            //         // ... prepare features ...
            //         let logits = model.forward(features);
            //         let probs = nn::loss::softmax(logits.clone(), 2);
            //         let top5 = probs.topk(5, 2);
            //         // Print top 5 words for first few positions
            //     }
            // }
        }

        let avg_loss = epoch_loss / train_data.len() as f32;
        let accuracy = (epoch_correct as f32 / epoch_total as f32) * 100.0;

        print!(
            "Epoch {}/{}: Loss = {:.4}, Acc = {:.1}%",
            epoch + 1,
            EPOCHS,
            avg_loss,
            accuracy
        );

        // Validation every 5 epochs
        if epoch % 5 == 0 {
            let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
            model.clone().save_file("model_temp", &recorder).unwrap();

            let inference_device: <InferenceBackend as Backend>::Device = Default::default();
            let val_model = WordWhisper::<InferenceBackend>::new(&config, &inference_device)
                .load_file("model_temp", &recorder, &inference_device)
                .unwrap();

            let mut val_loss = 0.0;
            let mut val_correct = 0;
            let mut val_total = 0;

            for (mel_frames, word_indices) in val_data.iter().take(50) {
                let features_flat: Vec<f32> = mel_frames.iter().flatten().copied().collect();
                let features = Tensor::<InferenceBackend, 3>::from_data(
                    TensorData::new(features_flat, [1, mel_frames.len(), N_MELS]),
                    &inference_device,
                );

                let mut padded_targets = vec![PAD_TOKEN; MAX_WORDS_PER_CLIP];
                for (i, &idx) in word_indices.iter().enumerate().take(MAX_WORDS_PER_CLIP) {
                    padded_targets[i] = idx;
                }

                let targets = Tensor::<InferenceBackend, 2, Int>::from_data(
                    TensorData::new(
                        padded_targets.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                        [1, MAX_WORDS_PER_CLIP],
                    ),
                    &inference_device,
                );

                let logits = val_model.forward(features);

                let predictions = logits.clone().argmax(2);
                let pred_data = predictions.into_data();
                let pred_indices = pred_data.as_slice::<i64>().unwrap();

                for i in 0..word_indices.len().min(MAX_WORDS_PER_CLIP) {
                    if pred_indices[i] == padded_targets[i] as i64 {
                        val_correct += 1;
                    }
                    val_total += 1;
                }

                let loss =
                    compute_loss_with_mask(logits, targets, PAD_TOKEN as i64, &inference_device);
                val_loss += loss.into_scalar();
            }

            let avg_val_loss = val_loss / val_data.len().min(50) as f32;
            let val_accuracy = (val_correct as f32 / val_total as f32) * 100.0;

            print!(
                ", Val Loss = {:.4}, Val Acc = {:.1}%",
                avg_val_loss, val_accuracy
            );

            if avg_val_loss < best_val_loss {
                best_val_loss = avg_val_loss;
                model.clone().save_file("model_best", &recorder).unwrap();
                print!(" 💾");
            }

            let _ = fs::remove_file("model_temp.mpk");
        }

        println!(", Time = {:.1}s", start_time.elapsed().as_secs_f32());

        if accuracy > 90.0 {
            println!("\n✨ Early stopping - high accuracy!");
            break;
        }
    }

    // Save final model
    let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file("model", &recorder)
        .expect("Failed to save model");

    println!("\n✅ Training complete!");
}

// Real-time transcription
fn transcribe_realtime() {
    println!("🎤 Loading model for transcription...");

    let device = Default::default();

    // Load vocabulary
    let vocab_json = fs::read_to_string("vocabulary.json")
        .expect("Vocabulary file not found. Train the model first!");
    let vocab: Vec<String> = serde_json::from_str(&vocab_json).unwrap();

    let config = WordWhisperConfig {
        vocab_size: vocab.len(),
        feature_dim: N_MELS,
        hidden_dim: HIDDEN_DIM,
        max_words: MAX_WORDS_PER_CLIP,
    };

    let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
    let model = WordWhisper::<InferenceBackend>::new(&config, &device)
        .load_file("model_best", &recorder, &device)
        .or_else(|_| {
            WordWhisper::<InferenceBackend>::new(&config, &device)
                .load_file("model", &recorder, &device)
        })
        .expect("No model found. Train first!");

    let audio_buffer = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let audio_buffer_clone = Arc::clone(&audio_buffer);

    let host = cpal::default_host();
    let input_device = host.default_input_device().expect("No input device");
    println!(
        "📢 Using device: {}",
        input_device.name().unwrap_or("Unknown".to_string())
    );

    let input_config = input_device
        .default_input_config()
        .expect("Failed to get config");
    let channels = input_config.channels();

    let stream = match input_config.sample_format() {
        cpal::SampleFormat::F32 => {
            input_device
                .build_input_stream(
                    &input_config.into(),
                    move |data: &[f32], _: &_| {
                        let mono: Vec<f32> = if channels > 1 {
                            data.chunks(channels as usize)
                                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                                .collect()
                        } else {
                            data.to_vec()
                        };

                        let mut buffer = audio_buffer_clone.lock().unwrap();
                        buffer.extend(mono.iter());

                        // Keep only last 15 seconds
                        while buffer.len() > SAMPLE_RATE as usize * 15 {
                            buffer.pop_front();
                        }
                    },
                    |err| eprintln!("Stream error: {}", err),
                    None,
                )
                .expect("Failed to build stream")
        }
        _ => panic!("Unsupported format"),
    };

    stream.play().expect("Failed to start stream");
    println!("\n🎯 Listening... (speak in complete sentences)\n");

    let mut last_transcription = String::new();

    loop {
        std::thread::sleep(std::time::Duration::from_millis(2000)); // Process every 2 seconds

        let buffer = audio_buffer.lock().unwrap();
        if buffer.len() >= SAMPLE_RATE as usize * 2 {
            // At least 2 seconds
            let audio: Vec<f32> = buffer.iter().copied().collect();
            drop(buffer);

            let rms = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();

            if rms > 0.01 {
                // If there's sound
                let mel_frames = compute_mel_spectrogram(&audio);

                let avg_energy: f32 = mel_frames
                    .iter()
                    .flat_map(|frame| frame.iter())
                    .map(|&x| x.abs())
                    .sum::<f32>()
                    / (mel_frames.len() * N_MELS) as f32;

                let target_energy = 2.0; // Typical training data energy
                let scale_factor = target_energy / (avg_energy + 1e-6);

                let normalized_frames: Vec<Vec<f32>> = mel_frames
                    .into_iter()
                    .map(|frame| frame.into_iter().map(|x| x * scale_factor).collect())
                    .collect();

                if normalized_frames.len() > 50 {
                    let features_flat: Vec<f32> =
                        normalized_frames.iter().flatten().copied().collect();
                    let features = Tensor::<InferenceBackend, 3>::from_data(
                        TensorData::new(features_flat, [1, normalized_frames.len(), N_MELS]),
                        &device,
                    );

                    let predictions = model.forward(features);
                    let text = decode_predictions(predictions, &vocab);

                    if !text.is_empty() && text != last_transcription {
                        println!("📝 {}", text);
                        last_transcription = text;
                    }
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("train") => {
            if let Some(dataset_path) = args.get(2) {
                train_model(dataset_path);
            } else {
                eprintln!("Usage: cargo run -- train <dataset_path>");
            }
        }
        Some("transcribe") => transcribe_realtime(),
        _ => {
            println!("Word-level Speech Recognition");
            println!("Usage:");
            println!("  cargo run -- train <dataset_path>");
            println!("  cargo run -- transcribe");
        }
    }
}
