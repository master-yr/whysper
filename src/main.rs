#![recursion_limit = "256"]

use burn::backend::{Autodiff, NdArray};
use burn::record::Record;
use burn::tensor::activation::softmax;
use burn::{
    config::Config,
    module::Module,
    nn,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{DefaultFileRecorder, FullPrecisionSettings},
    tensor::{Tensor, TensorData, activation::relu, backend::Backend},
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::WavReader;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::{HashMap, HashSet};
use std::fs::{self, read_to_string};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

// Backend selection
type TrainBackend = Autodiff<NdArray>;
type InferenceBackend = NdArray;

// Audio constants
const SAMPLE_RATE: u32 = 16000; // 16kHz for speech
const MAX_SAMPLES_PER_WORD: usize = 700;
const FRAME_SIZE: usize = 512; // FFT window size
const HOP_SIZE: usize = 160; // Step between frames
const N_MELS: usize = 80; // Mel frequency bins

// Model constants - SIMPLIFIED!
const HIDDEN_DIM: usize = 512; // Hidden layer size
const DROPOUT_RATE: f64 = 0.1; // Less aggressive dropout

// Training constants
const LEARNING_RATE: f64 = 0.0001;
const EPOCHS: usize = 50;
const TRAIN_SPLIT: f32 = 0.8;
const MAX_AUDIO_LENGTH: f32 = 2.0; // 2 seconds max per word

#[derive(Config)]
pub struct WordRecognizerConfig {
    vocab_size: usize,  // Number of words to recognize
    feature_dim: usize, // Input features (N_MELS)
    hidden_dim: usize,  // Hidden dimension
}

#[derive(Module, Debug)]
pub struct WordRecognizer<B: Backend> {
    // Feature extraction layers
    conv1: nn::conv::Conv1d<B>,
    bn1: nn::BatchNorm<B, 1>,
    conv2: nn::conv::Conv1d<B>,
    bn2: nn::BatchNorm<B, 1>,
    conv3: nn::conv::Conv1d<B>,
    bn3: nn::BatchNorm<B, 1>,

    // Global pooling to collapse time dimension
    global_pool: nn::pool::AdaptiveAvgPool1d,

    // Classification head
    classifier: nn::Linear<B>,
    dropout: nn::Dropout,
}

impl<B: Backend> WordRecognizer<B> {
    pub fn new(config: &WordRecognizerConfig, device: &B::Device) -> Self {
        println!("🔧 Creating Word Recognizer:");
        println!("   - Vocabulary: {} words", config.vocab_size);
        println!("   - Features: {} mel bins", config.feature_dim);
        println!("   - Hidden dim: {}", config.hidden_dim);

        // Convolutional layers for feature extraction
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

        // Global pooling reduces any time length to single vector
        let global_pool = nn::pool::AdaptiveAvgPool1dConfig::new(1).init();

        // Direct classification from pooled features
        let classifier = nn::LinearConfig::new(config.hidden_dim, config.vocab_size).init(device);
        let dropout = nn::DropoutConfig::new(DROPOUT_RATE).init();

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            global_pool,
            classifier,
            dropout,
        }
    }

    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, time, mels] = features.dims();

        // Reshape for conv layers: [batch, time, mels] -> [batch, mels, time]
        let x = features.swap_dims(1, 2);

        // Extract features through conv layers
        let x = relu(self.bn1.forward(self.conv1.forward(x)));
        let x = self.dropout.forward(x);

        let x = relu(self.bn2.forward(self.conv2.forward(x)));
        let x = self.dropout.forward(x);

        let x = relu(self.bn3.forward(self.conv3.forward(x)));

        // Global average pooling: [batch, hidden_dim, time] -> [batch, hidden_dim, 1]
        let pooled = self.global_pool.forward(x);

        // Squeeze out the time dimension: [batch, hidden_dim, 1] -> [batch, hidden_dim]
        let pooled = pooled.squeeze(2);

        // Classify: [batch, hidden_dim] -> [batch, vocab_size]
        self.classifier.forward(pooled)
    }
}

// Load audio file
fn load_audio(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = WavReader::open(path).map_err(|e| format!("Failed to open WAV: {}", e))?;

    let spec = reader.spec();
    if spec.channels != 1 {
        return Err("Expected mono audio".to_string());
    }
    if spec.sample_rate != SAMPLE_RATE {
        return Err(format!(
            "Expected {}Hz, got {}Hz",
            SAMPLE_RATE, spec.sample_rate
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

    let mel_filters = create_mel_filterbank();

    for i in (0..audio.len().saturating_sub(FRAME_SIZE)).step_by(HOP_SIZE) {
        let frame = &audio[i..i + FRAME_SIZE];

        // Apply Hann window
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

        // Power spectrum
        let power: Vec<f32> = fft_buffer[..FRAME_SIZE / 2]
            .iter()
            .map(|c| c.norm_sqr().max(1e-10))
            .collect();

        // Apply mel filterbank
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

// Create mel filterbank
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

// Build vocabulary from dataset folder
fn build_vocabulary(dataset_path: &str) -> (Vec<String>, HashMap<String, usize>) {
    println!("\n📚 Building vocabulary from dataset...");

    // let mut words = Vec::new();

    let mut words: Vec<String> = read_to_string("./vocabulary.json")
        .ok()
        .and_then(|e| serde_json::from_str(&e).ok())
        .unwrap_or(
            read_to_string(dataset_path.to_owned() + "/testing_list.txt")
                .unwrap()
                .lines()
                .into_iter()
                .map(|l| l.split('/').next().unwrap().to_owned())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect(),
        );

    // Sort for consistent ordering
    words.sort();

    println!("   Found {} unique words", words.len());
    println!("   Words: {:?}", &words);

    // Create word to index mapping
    let word_to_idx: HashMap<String, usize> = words
        .iter()
        .enumerate()
        .map(|(idx, word)| (word.clone(), idx))
        .collect();

    (words, word_to_idx)
}

// Load dataset
fn load_dataset(
    dataset_path: &str,
    word_to_idx: &HashMap<String, usize>,
) -> Vec<(Vec<Vec<f32>>, usize)> {
    println!("\n📁 Loading audio samples...");
    let mut data = Vec::new();
    let mut skipped = 0;
    let vocab: Vec<String> =
        serde_json::from_str(&read_to_string("./vocabulary.json").unwrap()).unwrap();

    for word in vocab.into_iter() {
        for (i, entry) in fs::read_dir(format!("{dataset_path}/{word}"))
            .unwrap()
            .enumerate()
        {
            if i > MAX_SAMPLES_PER_WORD {
                break;
            }
            let entry = entry.unwrap();
            let path = entry.path();

            // dbg!(&path);
            // dbg!(&entry);

            if path.extension().and_then(|s| s.to_str()) == Some("wav") {
                if let Some(&label) = word_to_idx.get(&word) {
                    match load_audio(&path) {
                        Ok(audio) => {
                            let duration = audio.len() as f32 / SAMPLE_RATE as f32;

                            if duration > MAX_AUDIO_LENGTH {
                                skipped += 1;
                                continue;
                            }

                            let mel_frames = compute_mel_spectrogram(&audio);

                            if mel_frames.len() > 10 {
                                // Minimum frames
                                data.push((mel_frames, label));

                                // if data.len() <= 5 {
                                print!(
                                    "\r   Loaded '{}' - {} frames, {:.2}s",
                                    word,
                                    data.last().unwrap().0.len(),
                                    duration
                                );
                                std::io::stdout().flush().unwrap();
                                // }
                            }
                        }
                        Err(e) => {
                            if data.is_empty() {
                                println!("   ⚠️  Error loading {}: {}", path.display(), e);
                            }
                            skipped += 1;
                        }
                    }
                }
            }
        }
    }

    println!("   Loaded {} samples, skipped {}", data.len(), skipped);
    data
}

// Training function
// Training function
fn train_model(dataset_path: &str) {
    println!("\n🚀 Starting Word Recognizer Training");

    let device = Default::default();

    // Build vocabulary
    let (vocab, word_to_idx) = build_vocabulary(dataset_path);

    if vocab.is_empty() {
        panic!("❌ No words found in dataset!");
    }

    // Save vocabulary
    let vocab_json = serde_json::to_string(&vocab).unwrap();
    fs::write("vocabulary.json", vocab_json).expect("Failed to save vocabulary");
    println!("💾 Saved vocabulary");

    let config = WordRecognizerConfig {
        vocab_size: vocab.len(),
        feature_dim: N_MELS,
        hidden_dim: HIDDEN_DIM,
    };

    let mut model = WordRecognizer::<TrainBackend>::new(&config, &device);
    let mut optimizer = AdamConfig::new().init();

    // Load and split dataset
    let mut all_data = load_dataset(dataset_path, &word_to_idx);
    if all_data.is_empty() {
        panic!("❌ No data loaded!");
    }

    let mut rng = thread_rng();
    all_data.shuffle(&mut rng);

    let split_idx = (all_data.len() as f32 * TRAIN_SPLIT) as usize;
    let (train_data, val_data) = all_data.split_at(split_idx);

    println!("\n🎯 Training Configuration:");
    println!("   Train samples: {}", train_data.len());
    println!("   Val samples: {}", val_data.len());
    println!("   Vocabulary size: {}", vocab.len());
    println!("   Learning rate: {}", LEARNING_RATE);

    let mut best_val_acc = 0.0;
    let training_start = std::time::Instant::now();
    let mut epoch_times = Vec::new();

    // Training loop
    for epoch in 0..EPOCHS {
        let epoch_start = std::time::Instant::now();
        println!("\n📊 Epoch {}/{}", epoch + 1, EPOCHS);

        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0;
        let mut confusion_matrix: HashMap<(String, String), usize> = HashMap::new();

        // Shuffle training data
        let mut train_batch: Vec<_> = train_data.to_vec();
        train_batch.shuffle(&mut rng);

        // Show progress bar
        let total_samples = train_batch.len();
        let show_every = (total_samples / 20).max(1); // Show ~20 updates per epoch

        for (idx, (mel_frames, label)) in train_batch.iter().enumerate() {
            // Prepare input tensor
            let features_flat: Vec<f32> = mel_frames.iter().flatten().copied().collect();
            let features = Tensor::<TrainBackend, 3>::from_data(
                TensorData::new(features_flat, [1, mel_frames.len(), N_MELS]),
                &device,
            );

            // Prepare target
            let target = Tensor::<TrainBackend, 1, burn::tensor::Int>::from_data(
                TensorData::new(vec![*label as i64], [1]),
                &device,
            );

            // Forward pass
            let logits = model.forward(features);

            // Check prediction
            let pred = logits.clone().argmax(1);
            let pred_idx = pred.into_data().as_slice::<i64>().unwrap()[0] as usize;

            let actual_word = &vocab[*label];
            let predicted_word = &vocab[pred_idx];

            if pred_idx == *label {
                epoch_correct += 1;
            }

            // Track confusion
            *confusion_matrix
                .entry((actual_word.clone(), predicted_word.clone()))
                .or_insert(0) += 1;

            // Compute loss
            let loss = nn::loss::CrossEntropyLossConfig::new()
                .init(&device)
                .forward(logits, target);

            let loss_value = loss.clone().into_scalar();
            epoch_loss += loss_value;

            // Show sample predictions periodically
            if idx % show_every == 0 || idx < 5 {
                let progress = (idx + 1) as f32 / total_samples as f32 * 100.0;
                let current_acc = epoch_correct as f32 / (idx + 1) as f32 * 100.0;
                let avg_loss = epoch_loss / (idx + 1) as f32;

                println!(
                    "   [{:>3.0}%] Loss: {:.4}, Acc: {:.1}% | '{}' → '{}' {}",
                    progress,
                    avg_loss,
                    current_acc,
                    actual_word,
                    predicted_word,
                    if pred_idx == *label { "✓" } else { "✗" }
                );
                // std::io::stdout().flush().unwrap();
            }

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(LEARNING_RATE, model, grads);
        }

        let train_acc = epoch_correct as f32 / train_data.len() as f32 * 100.0;
        let avg_loss = epoch_loss / train_data.len() as f32;

        // Clear progress line and show final stats
        print!("\r");
        println!(
            "   ✅ Train - Loss: {:.4}, Accuracy: {:.1}%",
            avg_loss, train_acc
        );

        // Show top confusions
        let mut confusions: Vec<_> = confusion_matrix
            .into_iter()
            .filter(|((a, p), _)| a != p)
            .collect();
        confusions.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        if !confusions.is_empty() && epoch % 5 == 0 {
            println!("   📊 Top confusions:");
            for ((actual, predicted), count) in confusions.iter().take(3) {
                println!("      '{}' → '{}' ({} times)", actual, predicted, count);
            }
        }

        // Validation
        if epoch % 5 == 4 || epoch == 0 {
            println!("   🔍 Validating...");

            // Save and reload for inference
            let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
            model.clone().save_file("model_temp", &recorder).unwrap();

            let inference_device: <InferenceBackend as Backend>::Device = Default::default();
            let val_model = WordRecognizer::<InferenceBackend>::new(&config, &inference_device)
                .load_file("model_temp", &recorder, &inference_device)
                .unwrap();

            let mut val_correct = 0;
            let mut val_loss = 0.0;

            for (mel_frames, label) in val_data.iter() {
                let features_flat: Vec<f32> = mel_frames.iter().flatten().copied().collect();
                let features = Tensor::<InferenceBackend, 3>::from_data(
                    TensorData::new(features_flat, [1, mel_frames.len(), N_MELS]),
                    &inference_device,
                );

                let target = Tensor::<InferenceBackend, 1, burn::tensor::Int>::from_data(
                    TensorData::new(vec![*label as i64], [1]),
                    &inference_device,
                );

                let logits = val_model.forward(features);
                let pred = logits.clone().argmax(1);
                let pred_idx = pred.into_data().as_slice::<i64>().unwrap()[0] as usize;

                if pred_idx == *label {
                    val_correct += 1;
                }

                let loss = nn::loss::CrossEntropyLossConfig::new()
                    .init(&inference_device)
                    .forward(logits, target);
                val_loss += loss.into_scalar();
            }

            let val_acc = val_correct as f32 / val_data.len() as f32 * 100.0;
            let avg_val_loss = val_loss / val_data.len() as f32;
            println!(
                "   📈 Val - Loss: {:.4}, Accuracy: {:.1}%",
                avg_val_loss, val_acc
            );

            // Save best model
            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                model.clone().save_file("model_best", &recorder).unwrap();
                println!("   💾 New best model saved! (prev: {:.1}%)", best_val_acc);
            }

            let _ = fs::remove_file("model_temp.mpk");
        }

        // Calculate and show ETA
        let epoch_time = epoch_start.elapsed();
        epoch_times.push(epoch_time);

        let avg_epoch_time =
            epoch_times.iter().sum::<std::time::Duration>() / epoch_times.len() as u32;
        let remaining_epochs = EPOCHS - epoch - 1;
        let eta = avg_epoch_time * remaining_epochs as u32;

        println!(
            "   ⏱️  Epoch time: {:.1}s | ETA: {}m {}s",
            epoch_time.as_secs_f32(),
            eta.as_secs() / 60,
            eta.as_secs() % 60
        );
    }

    // Save final model
    let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file("model", &recorder)
        .expect("Failed to save model");

    let total_time = training_start.elapsed();
    println!("\n✅ Training complete!");
    println!("   Best validation accuracy: {:.1}%", best_val_acc);
    println!(
        "   Total training time: {}m {}s",
        total_time.as_secs() / 60,
        total_time.as_secs() % 60
    );
}

// Real-time recognition
fn recognize_realtime() {
    println!("🎤 Loading model for recognition...");

    let device = Default::default();

    // Load vocabulary
    let vocab_json =
        fs::read_to_string("vocabulary.json").expect("Vocabulary not found. Train first!");
    let vocab: Vec<String> = serde_json::from_str(&vocab_json).unwrap();

    let config = WordRecognizerConfig {
        vocab_size: vocab.len(),
        feature_dim: N_MELS,
        hidden_dim: HIDDEN_DIM,
    };

    let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
    let model = WordRecognizer::<InferenceBackend>::new(&config, &device)
        .load_file("model_best", &recorder, &device)
        .or_else(|_| {
            WordRecognizer::<InferenceBackend>::new(&config, &device)
                .load_file("model", &recorder, &device)
        })
        .expect("No model found!");

    println!("✅ Model loaded - recognizing {} words:", vocab.len());
    for (i, word) in vocab.iter().enumerate() {
        print!("{:>15}", word);
        if (i + 1) % 5 == 0 {
            println!();
        }
    }
    println!("\n");

    // Audio buffer
    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let audio_buffer_clone = Arc::clone(&audio_buffer);
    let is_recording = Arc::new(Mutex::new(false));
    let is_recording_clone = Arc::clone(&is_recording);

    // Setup audio
    let host = cpal::default_host();
    let input_device = host.default_input_device().expect("No input device");
    let input_config = input_device
        .default_input_config()
        .expect("Failed to get config");
    let channels = input_config.channels();

    let stream = match input_config.sample_format() {
        cpal::SampleFormat::F32 => input_device
            .build_input_stream(
                &input_config.into(),
                move |data: &[f32], _: &_| {
                    if *is_recording_clone.lock().unwrap() {
                        let mono: Vec<f32> = if channels > 1 {
                            data.chunks(channels as usize)
                                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                                .collect()
                        } else {
                            data.to_vec()
                        };

                        audio_buffer_clone.lock().unwrap().extend(mono);
                    }
                },
                |err| eprintln!("Stream error: {}", err),
                None,
            )
            .expect("Failed to build stream"),
        _ => panic!("Unsupported format"),
    };

    stream.play().expect("Failed to start stream");

    println!("🎯 Press ENTER to record a word, Ctrl+C to quit\n");

    loop {
        // Wait for enter
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        // Clear buffer and start recording
        audio_buffer.lock().unwrap().clear();
        *is_recording.lock().unwrap() = true;

        println!("🔴 Recording... (say one word clearly)");
        std::thread::sleep(std::time::Duration::from_millis(1500));

        // Stop recording
        *is_recording.lock().unwrap() = false;
        let audio = audio_buffer.lock().unwrap().clone();

        if audio.len() > SAMPLE_RATE as usize / 4 {
            // At least 0.25 seconds
            println!("🔊 Processing {} samples...", audio.len());

            let mel_frames = compute_mel_spectrogram(&audio);

            if mel_frames.len() > 10 {
                let features_flat: Vec<f32> = mel_frames.iter().flatten().copied().collect();
                let features = Tensor::<InferenceBackend, 3>::from_data(
                    TensorData::new(features_flat, [1, mel_frames.len(), N_MELS]),
                    &device,
                );

                let logits = model.forward(features);
                let probs = softmax(logits, 1);
                let pred = probs.clone().argmax(1);

                let pred_idx = pred.into_data().as_slice::<i64>().unwrap()[0] as usize;
                let confidence = probs.into_data().as_slice::<f32>().unwrap()[pred_idx];

                println!(
                    "\n✨ Recognized: '{}' (confidence: {:.1}%)\n",
                    vocab[pred_idx],
                    confidence * 100.0
                );
            } else {
                println!("❌ Too short, try again\n");
            }
        } else {
            println!("❌ No audio captured\n");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    println!("\n🎙️  Word Recognizer - Single Word Recognition");

    match args.get(1).map(|s| s.as_str()) {
        Some("train") => {
            if let Some(dataset_path) = args.get(2) {
                if !Path::new(dataset_path).exists() {
                    eprintln!("❌ Dataset path '{}' not found!", dataset_path);
                    std::process::exit(1);
                }
                train_model(dataset_path);
            } else {
                eprintln!("Usage: cargo run -- train <dataset_path>");
            }
        }
        Some("recognize") => {
            recognize_realtime();
        }
        _ => {
            println!("Usage:");
            println!("  cargo run -- train <dataset_path>    Train the model");
            println!("  cargo run -- recognize               Real-time recognition");
            println!("\nDataset format:");
            println!("  dataset/");
            println!("    apple.wav    (recording of 'apple')");
            println!("    banana.wav   (recording of 'banana')");
            println!("    hello.wav    (recording of 'hello')");
            println!("    ...");
            println!("\nEach WAV file should be:");
            println!("  - 16kHz, mono");
            println!("  - Named after the word it contains");
            println!("  - Under 2 seconds long");
        }
    }
}

// #![recursion_limit = "256"]

// use burn::backend::{Autodiff, NdArray};
// use burn::{
//     config::Config,
//     module::Module,
//     nn,
//     optim::{AdamConfig, GradientsParams, Optimizer},
//     record::{DefaultFileRecorder, FullPrecisionSettings},
//     tensor::{Int, Tensor, TensorData, activation::relu, backend::Backend},
// };
// use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
// use hound::WavReader;
// use rand::seq::SliceRandom;
// use rand::thread_rng;
// use rustfft::{FftPlanner, num_complex::Complex};
// use std::collections::{HashMap, VecDeque};
// use std::fs;
// use std::io::Write;
// use std::path::Path;
// use std::sync::{Arc, Mutex};

// // Backend selection - NdArray is CPU-based, good for debugging
// // Wgpu is GPU-based, faster but harder to debug
// type TrainBackend = Autodiff<NdArray>;
// type InferenceBackend = NdArray;
// // type TrainBackend = Autodiff<Wgpu>;
// // type InferenceBackend = Wgpu;

// // Audio constants
// const SAMPLE_RATE: u32 = 16000; // 16kHz - standard for speech recognition
// const FRAME_SIZE: usize = 512; // FFT window size (~32ms at 16kHz)
// const HOP_SIZE: usize = 160; // Step size between frames (~10ms)

// // Model constants - ADJUSTED FOR 3-SECOND CLIPS
// const MAX_WORDS_PER_CLIP: usize = 10; // 3 seconds typically contains 5-10 words, 15 allows headroom
// const WORD_VOCAB_SIZE: usize = 70; // Smaller vocab = easier to learn initially
// const HIDDEN_DIM: usize = 256; // LSTM hidden dimension - balance between capacity and speed
// const N_MELS: usize = 160; // Number of mel-frequency bins (standard for speech)
// const LSTM_LAYERS: usize = 3; // Stacked LSTM layers for better temporal modeling

// // Training constants - IMPROVED
// const TRAIN_SPLIT: f32 = 0.8; // 80% train, 20% validation
// const INITIAL_LR: f64 = 0.0001; // Starting learning rate
// const MIN_LR: f64 = 0.00001; // Minimum learning rate floor
// const LR_DECAY_RATE: f64 = 0.95; // Multiply LR by this each epoch
// const EPOCHS: usize = 100; // Maximum training epochs
// const MAX_AUDIO_LENGTH: f32 = 3.0; // Maximum audio clip length in seconds
// const MIN_WORD_COUNT: usize = 2; // Minimum words per clip (filters out very short utterances)
// const BATCH_SIZE: usize = 16; // Batch size (not fully implemented yet)
// const DROPOUT_RATE: f64 = 0.5; // Dropout for regularization (50% is aggressive)

// const LEARNING_RATE: f64 = 0.0001; // Fixed learning rate (overrides dynamic LR for now)

// // Special tokens - CRITICAL for sequence modeling
// const PAD_TOKEN: usize = 0; // Padding for shorter sequences
// const UNK_TOKEN: usize = 1; // Unknown/out-of-vocabulary words
// const SOS_TOKEN: usize = 2; // Start-of-sequence marker
// const EOS_TOKEN: usize = 3; // End-of-sequence marker

// #[derive(Config)]
// pub struct WordWhisperConfig {
//     vocab_size: usize,  // Total vocabulary size including special tokens
//     feature_dim: usize, // Input feature dimension (N_MELS)
//     hidden_dim: usize,  // Hidden layer dimension
//     max_words: usize,   // Maximum words per sequence
// }

// #[derive(Module, Debug)]
// pub struct WordWhisper<B: Backend> {
//     // Enhanced audio encoder - 3 conv layers with batch norm
//     // Conv layers extract local patterns from mel-spectrograms
//     conv1: nn::conv::Conv1d<B>, // 80 -> 128 channels
//     bn1: nn::BatchNorm<B, 1>,   // Batch normalization for stable training
//     conv2: nn::conv::Conv1d<B>, // 128 -> 256 channels
//     bn2: nn::BatchNorm<B, 1>,
//     conv3: nn::conv::Conv1d<B>, // 256 -> hidden_dim channels
//     bn3: nn::BatchNorm<B, 1>,
//     pool: nn::pool::AdaptiveAvgPool1d, // Pools variable length to fixed MAX_WORDS_PER_CLIP

//     // Stacked LSTM layers for temporal modeling
//     lstm1: nn::lstm::Lstm<B>, // First LSTM layer
//     lstm2: nn::lstm::Lstm<B>, // Second LSTM layer (deeper = better context)

//     // Word prediction head with intermediate layer
//     pre_proj: nn::Linear<B>,  // hidden_dim -> hidden_dim/2 (bottleneck)
//     word_proj: nn::Linear<B>, // hidden_dim/2 -> vocab_size (final predictions)
//     dropout: nn::Dropout,     // Dropout for regularization
// }

// impl<B: Backend> WordWhisper<B> {
//     pub fn new(config: &WordWhisperConfig, device: &B::Device) -> Self {
//         println!("🔧 Creating WordWhisper model:");
//         println!("   - Vocab size: {}", config.vocab_size);
//         println!("   - Feature dim: {}", config.feature_dim);
//         println!("   - Hidden dim: {}", config.hidden_dim);
//         println!("   - Max words: {}", config.max_words);

//         // Larger kernels (size 5) for better temporal modeling
//         // Example: kernel size 5 means looking at 5 consecutive time frames
//         let conv1 = nn::conv::Conv1dConfig::new(config.feature_dim, 128, 5)
//             .with_padding(nn::PaddingConfig1d::Same) // Same padding preserves sequence length
//             .init(device);
//         let bn1 = nn::BatchNormConfig::new(128).init(device);

//         let conv2 = nn::conv::Conv1dConfig::new(128, 256, 5)
//             .with_padding(nn::PaddingConfig1d::Same)
//             .init(device);
//         let bn2 = nn::BatchNormConfig::new(256).init(device);

//         let conv3 = nn::conv::Conv1dConfig::new(256, config.hidden_dim, 5)
//             .with_padding(nn::PaddingConfig1d::Same)
//             .init(device);
//         let bn3 = nn::BatchNormConfig::new(config.hidden_dim).init(device);

//         // Adaptive pooling reduces variable length sequences to fixed MAX_WORDS_PER_CLIP
//         let pool = nn::pool::AdaptiveAvgPool1dConfig::new(config.max_words).init();

//         // Stacked LSTMs - bidirectional=false for causal modeling
//         let lstm1 =
//             nn::lstm::LstmConfig::new(config.hidden_dim, config.hidden_dim, false).init(device);
//         let lstm2 =
//             nn::lstm::LstmConfig::new(config.hidden_dim, config.hidden_dim, false).init(device);

//         // Two-layer prediction head with bottleneck
//         // Example: 512 -> 256 -> vocab_size
//         let pre_proj = nn::LinearConfig::new(config.hidden_dim, config.hidden_dim / 2).init(device);
//         let word_proj =
//             nn::LinearConfig::new(config.hidden_dim / 2, config.vocab_size).init(device);
//         let dropout = nn::DropoutConfig::new(DROPOUT_RATE).init();

//         Self {
//             conv1,
//             bn1,
//             conv2,
//             bn2,
//             conv3,
//             bn3,
//             pool,
//             lstm1,
//             lstm2,
//             pre_proj,
//             word_proj,
//             dropout,
//         }
//     }

//     pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
//         let [batch, time, mels] = features.dims();

//         // DEBUG: Print input shape
//         if batch == 1 && time < 100 {
//             // Only print for small batches to avoid spam
//             println!(
//                 "🔍 Forward pass - Input shape: [{}, {}, {}]",
//                 batch, time, mels
//             );
//         }

//         // Conv expects [batch, channels, time] so we swap dims
//         // Example: [1, 300, 80] -> [1, 80, 300]
//         let x = features.swap_dims(1, 2);

//         // Conv block 1: Extract low-level features
//         let x = relu(self.bn1.forward(self.conv1.forward(x)));
//         let x = self.dropout.forward(x); // Dropout after activation

//         // Conv block 2: Extract mid-level features
//         let x = relu(self.bn2.forward(self.conv2.forward(x)));
//         let x = self.dropout.forward(x);

//         // Conv block 3: Extract high-level features
//         let x = relu(self.bn3.forward(self.conv3.forward(x)));

//         // Pool to fixed length MAX_WORDS_PER_CLIP
//         // Example: [1, 512, 300] -> [1, 512, 15]
//         let x = self.pool.forward(x);

//         // Swap back for LSTM: [batch, channels, time] -> [batch, time, channels]
//         let x = x.swap_dims(1, 2);

//         // LSTM expects [time, batch, features] so swap again
//         let x_t = x.swap_dims(0, 1);

//         // First LSTM layer
//         let (x, _) = self.lstm1.forward(x_t, None);
//         let x = self.dropout.forward(x); // Dropout between LSTM layers

//         // Second LSTM layer
//         let (output, _) = self.lstm2.forward(x, None);

//         // Swap back to [batch, time, features]
//         let output = output.swap_dims(0, 1);

//         // Two-layer prediction head with ReLU and dropout
//         let output = self.dropout.forward(output);
//         let output = relu(self.pre_proj.forward(output));
//         let output = self.dropout.forward(output);
//         let output = self.word_proj.forward(output);

//         // DEBUG: Print output shape
//         if batch == 1 && time < 100 {
//             let [b, t, v] = output.dims();
//             println!("🔍 Forward pass - Output shape: [{}, {}, {}]", b, t, v);
//         }

//         output
//     }
// }

// // Build vocabulary from dataset
// fn build_vocabulary(dataset_path: &str) -> (Vec<String>, HashMap<String, usize>) {
//     println!("\n📚 Building vocabulary from dataset...");
//     println!("   Dataset path: {}", dataset_path);

//     let mut word_counts: HashMap<String, usize> = HashMap::new();
//     let mut total_files = 0;
//     let mut total_words = 0;

//     // Count words in all transcripts
//     for entry in fs::read_dir(dataset_path).unwrap() {
//         let entry = entry.unwrap();
//         let path = entry.path();

//         if path.extension().and_then(|s| s.to_str()) == Some("txt") {
//             total_files += 1;

//             if let Ok(text) = fs::read_to_string(&path) {
//                 // Example text: "hello world how are you"
//                 for word in text.to_lowercase().split_whitespace() {
//                     // Clean word - remove punctuation
//                     // Example: "hello," -> "hello"
//                     let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
//                     if !clean_word.is_empty() {
//                         *word_counts.entry(clean_word.to_string()).or_insert(0) += 1;
//                         total_words += 1;
//                     }
//                 }
//             }

//             if total_files % 100 == 0 {
//                 println!("   Processed {} transcript files...", total_files);
//             }
//         }
//     }

//     println!("   Total transcript files: {}", total_files);
//     println!("   Total word occurrences: {}", total_words);
//     println!("   Unique words found: {}", word_counts.len());

//     // Sort by frequency and take top words
//     let mut words: Vec<(String, usize)> = word_counts.into_iter().collect();
//     words.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by count

//     // Create vocabulary with special tokens
//     let mut vocab = vec![
//         "<pad>".to_string(), // Index 0: padding token
//         "<unk>".to_string(), // Index 1: unknown word token
//         "<sos>".to_string(), // Index 2: start of sequence
//         "<eos>".to_string(), // Index 3: end of sequence
//     ];

//     // Add top frequent words
//     println!("\n   Top 20 most frequent words:");
//     for (word, count) in words.iter().take(WORD_VOCAB_SIZE - 4) {
//         vocab.push(word.clone());
//         if vocab.len() <= 24 {
//             // Show top 20 regular words
//             println!("     {:3}. '{}' (count: {})", vocab.len() - 4, word, count);
//         }
//     }

//     // Create word to index mapping
//     // Example: {"hello": 5, "world": 6, ...}
//     let word_to_idx: HashMap<String, usize> = vocab
//         .iter()
//         .enumerate()
//         .map(|(idx, word)| (word.clone(), idx))
//         .collect();

//     println!("\n✅ Vocabulary built successfully:");
//     println!("   - Total vocabulary size: {}", vocab.len());
//     println!("   - Regular words: {}", vocab.len() - 4);
//     println!("   - Special tokens: 4");

//     (vocab, word_to_idx)
// }

// // Load and preprocess audio
// fn load_audio(path: &Path) -> Result<Vec<f32>, String> {
//     let mut reader = WavReader::open(path).map_err(|e| format!("Failed to open WAV: {}", e))?;

//     let spec = reader.spec();

//     // Validate audio format
//     if spec.channels != 1 {
//         return Err(format!(
//             "Expected mono audio (1 channel), got {} channels. File: {:?}",
//             spec.channels, path
//         ));
//     }

//     if spec.sample_rate != SAMPLE_RATE {
//         return Err(format!(
//             "Expected {}Hz sample rate, got {}Hz. File: {:?}",
//             SAMPLE_RATE, spec.sample_rate, path
//         ));
//     }

//     // Convert i16 samples to f32 normalized to [-1, 1]
//     // Example: 16384 (i16) -> 0.5 (f32)
//     let samples: Vec<f32> = reader
//         .samples::<i16>()
//         .map(|s| s.unwrap() as f32 / 32768.0)
//         .collect();

//     Ok(samples)
// }

// // Compute mel spectrogram
// fn compute_mel_spectrogram(audio: &[f32]) -> Vec<Vec<f32>> {
//     let mut planner = FftPlanner::new();
//     let fft = planner.plan_fft_forward(FRAME_SIZE);
//     let mut mel_frames = Vec::new();

//     // Create mel filterbank (triangular filters in mel scale)
//     let mel_filters = create_mel_filterbank();

//     // Process audio in overlapping frames
//     // Example: 16000 samples with hop=160 -> ~100 frames
//     let _total_frames = (audio.len().saturating_sub(FRAME_SIZE)) / HOP_SIZE + 1;

//     for (frame_idx, i) in (0..audio.len().saturating_sub(FRAME_SIZE))
//         .step_by(HOP_SIZE)
//         .enumerate()
//     {
//         let frame = &audio[i..i + FRAME_SIZE];

//         // Apply Hann window to reduce spectral leakage
//         // Window formula: 0.5 - 0.5 * cos(2π * n / (N-1))
//         let windowed: Vec<Complex<f32>> = frame
//             .iter()
//             .enumerate()
//             .map(|(j, &sample)| {
//                 let window = 0.5
//                     - 0.5 * (2.0 * std::f32::consts::PI * j as f32 / (FRAME_SIZE - 1) as f32).cos();
//                 Complex::new(sample * window, 0.0)
//             })
//             .collect();

//         let mut fft_buffer = windowed;
//         fft.process(&mut fft_buffer);

//         // Convert complex FFT output to power spectrum
//         // Only use first half (Nyquist frequency)
//         let power: Vec<f32> = fft_buffer[..FRAME_SIZE / 2]
//             .iter()
//             .map(|c| c.norm_sqr().max(1e-10)) // Avoid log(0)
//             .collect();

//         // Apply mel filterbank and convert to log scale
//         let mut mel_frame = vec![0.0; N_MELS];
//         for (i, filter) in mel_filters.iter().enumerate() {
//             // Dot product of power spectrum with mel filter
//             mel_frame[i] = power
//                 .iter()
//                 .zip(filter.iter())
//                 .map(|(p, f)| p * f)
//                 .sum::<f32>()
//                 .log10(); // Log scale for perceptual relevance
//         }

//         // Debug first few frames
//         if frame_idx < 3 {
//             let energy: f32 = mel_frame.iter().map(|x| x.abs()).sum();
//             println!("   Frame {}: energy = {:.2}", frame_idx, energy);
//         }

//         mel_frames.push(mel_frame);
//     }

//     println!(
//         "   Computed {} mel frames from {} audio samples",
//         mel_frames.len(),
//         audio.len()
//     );
//     mel_frames
// }

// // Simple mel filterbank
// fn create_mel_filterbank() -> Vec<Vec<f32>> {
//     let mut filterbank = vec![vec![0.0; FRAME_SIZE / 2]; N_MELS];

//     // Create triangular filters evenly spaced in mel scale
//     // Example: 80 filters covering 0-8000Hz
//     for i in 0..N_MELS {
//         let center = (i + 1) * (FRAME_SIZE / 2) / (N_MELS + 1);
//         let width = FRAME_SIZE / (2 * N_MELS);

//         for j in 0..FRAME_SIZE / 2 {
//             if j >= center.saturating_sub(width) && j <= center + width {
//                 // Triangular filter: peaks at center, falls to 0 at edges
//                 let distance = (j as i32 - center as i32).abs() as f32;
//                 filterbank[i][j] = 1.0 - (distance / width as f32);
//             }
//         }
//     }

//     filterbank
// }

// // Convert text to word indices
// fn text_to_word_indices(text: &str, word_to_idx: &HashMap<String, usize>) -> Vec<usize> {
//     let mut indices = vec![SOS_TOKEN]; // Always start with SOS

//     // Example: "hello world" -> [2, 5, 6, 3] (SOS, hello_idx, world_idx, EOS)
//     for word in text.to_lowercase().split_whitespace() {
//         let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
//         if !clean_word.is_empty() {
//             // Use UNK_TOKEN if word not in vocabulary
//             let idx = word_to_idx.get(clean_word).unwrap_or(&UNK_TOKEN);
//             indices.push(*idx);
//         }
//     }

//     indices.push(EOS_TOKEN); // Always end with EOS
//     indices
// }

// // Decode predictions back to text
// fn decode_predictions(predictions: Tensor<InferenceBackend, 3>, vocab: &[String]) -> String {
//     // Get most likely word at each position
//     let pred_data = predictions.argmax(2).into_data();
//     let indices = pred_data.as_slice::<i64>().unwrap();

//     println!("\n🔍 Decoding predictions:");
//     println!("   Raw indices: {:?}", &indices[..indices.len().min(10)]);

//     let mut words = Vec::new();
//     for (pos, &idx) in indices.iter().enumerate() {
//         if idx as usize >= vocab.len() {
//             println!(
//                 "   ⚠️  Position {}: Invalid index {} (vocab size: {})",
//                 pos,
//                 idx,
//                 vocab.len()
//             );
//             continue;
//         }

//         let word = &vocab[idx as usize];
//         println!("   Position {}: {} -> '{}'", pos, idx, word);

//         // Skip special tokens and consecutive duplicates
//         if word != "<pad>" && word != "<sos>" && word != "<eos>" {
//             if words.is_empty() || words.last() != Some(word) {
//                 words.push(word.clone());
//             }
//         }
//     }

//     let result = words.join(" ");
//     println!("   Final decoded text: '{}'", result);
//     result
// }

// // Load dataset
// fn load_dataset(
//     dataset_path: &str,
//     word_to_idx: &HashMap<String, usize>,
// ) -> Vec<(Vec<Vec<f32>>, Vec<usize>)> {
//     println!("\n📁 Loading dataset from: {}", dataset_path);
//     let mut data = Vec::new();
//     let mut skipped = 0;
//     let mut skipped_reasons = HashMap::new();

//     for entry in fs::read_dir(dataset_path).unwrap() {
//         let entry = entry.unwrap();
//         let path = entry.path();

//         if path.extension().and_then(|s| s.to_str()) == Some("wav") {
//             let txt_path = path.with_extension("txt");

//             if txt_path.exists() {
//                 match load_audio(&path) {
//                     Ok(audio) => {
//                         if let Ok(text) = fs::read_to_string(&txt_path) {
//                             // Check audio length
//                             let duration = audio.len() as f32 / SAMPLE_RATE as f32;
//                             if duration > MAX_AUDIO_LENGTH {
//                                 *skipped_reasons.entry("too_long").or_insert(0) += 1;
//                                 skipped += 1;
//                                 continue;
//                             }

//                             let word_indices = text_to_word_indices(&text, word_to_idx);

//                             // Skip if too few words (need at least MIN_WORD_COUNT + SOS + EOS)
//                             if word_indices.len() < MIN_WORD_COUNT + 2 {
//                                 *skipped_reasons.entry("too_few_words").or_insert(0) += 1;
//                                 skipped += 1;
//                                 continue;
//                             }

//                             // Skip if too many words
//                             if word_indices.len() > MAX_WORDS_PER_CLIP {
//                                 *skipped_reasons.entry("too_many_words").or_insert(0) += 1;
//                                 skipped += 1;
//                                 continue;
//                             }

//                             let mel_frames = compute_mel_spectrogram(&audio);

//                             if mel_frames.len() < 10 {
//                                 // Too short
//                                 *skipped_reasons.entry("too_few_frames").or_insert(0) += 1;
//                                 skipped += 1;
//                                 continue;
//                             }

//                             // Successfully loaded sample
//                             let word_count = word_indices.len() - 2; // Exclude SOS/EOS
//                             let frame_count = mel_frames.len();

//                             data.push((mel_frames, word_indices.clone()));

//                             // Debug info for first few samples
//                             if data.len() <= 5 {
//                                 println!("\n  📄 Sample {}:", data.len());
//                                 println!(
//                                     "     File: {}",
//                                     path.file_name().unwrap().to_str().unwrap()
//                                 );
//                                 println!("     Text: '{}'", text.trim());
//                                 println!("     Word indices: {:?}", word_indices);
//                                 println!("     Duration: {:.2}s", duration);
//                                 println!("     Words: {} (excluding SOS/EOS)", word_count);
//                                 println!("     Mel frames: {}", frame_count);
//                             }
//                         }
//                     }
//                     Err(e) => {
//                         if data.is_empty() {
//                             // Only print first error
//                             println!("  ⚠️  Error loading {}: {}", path.display(), e);
//                         }
//                         *skipped_reasons.entry("load_error").or_insert(0) += 1;
//                         skipped += 1;
//                     }
//                 }
//             }
//         }

//         if data.len() % 100 == 0 && data.len() > 0 {
//             print!("\r  Loaded {} samples...", data.len());
//             std::io::stdout().flush().unwrap();
//         }
//     }

//     println!("\n\n✅ Dataset loading complete:");
//     println!("   - Successfully loaded: {} samples", data.len());
//     println!("   - Skipped: {} samples", skipped);

//     if !skipped_reasons.is_empty() {
//         println!("\n   Skipped reasons:");
//         for (reason, count) in skipped_reasons {
//             println!("     - {}: {}", reason, count);
//         }
//     }

//     data
// }

// // Custom loss function that ignores padding
// fn compute_loss_with_mask<B: Backend>(
//     logits: Tensor<B, 3>,
//     targets: Tensor<B, 2, Int>,
//     pad_token: i64,
//     actual_lengths: Vec<usize>, // Add actual sequence lengths
//     device: &B::Device,
// ) -> Tensor<B, 1> {
//     let [batch, seq_len, vocab_size] = logits.dims();

//     // Debug shapes
//     println!("\n🔍 Loss computation:");
//     println!("   Logits shape: [{}, {}, {}]", batch, seq_len, vocab_size);
//     println!("   Targets shape: {:?}", targets.dims());
//     println!("   Actual lengths: {:?}", actual_lengths);

//     // Reshape for cross entropy: [batch*seq_len, vocab_size]
//     let logits_flat = logits.reshape([batch * seq_len, vocab_size]);
//     let targets_flat = targets.reshape([batch * seq_len]);

//     // Create length-based mask to ignore everything after actual sequence
//     let mut mask_data = vec![0.0f32; batch * seq_len];
//     for (b, &length) in actual_lengths.iter().enumerate() {
//         for i in 0..length.min(seq_len) {
//             mask_data[b * seq_len + i] = 1.0;
//         }
//     }

//     let mask = Tensor::<B, 1>::from_data(TensorData::new(mask_data, [batch * seq_len]), device);
//     // Count non-padding tokens
//     let non_pad_count = mask.clone().sum().into_scalar();
//     println!(
//         "   Non-padding tokens: {} / {}",
//         non_pad_count,
//         batch * seq_len
//     );

//     // Compute cross entropy loss
//     let loss = nn::loss::CrossEntropyLossConfig::new()
//         .init(device)
//         .forward(logits_flat, targets_flat);

//     // Apply mask: only count loss for non-padding positions
//     let masked_loss = loss * mask.clone();
//     let sum_loss = masked_loss.sum();
//     let count = mask.sum().add_scalar(1e-6); // Avoid division by zero

//     let avg_loss = sum_loss.div(count).reshape([1]);

//     println!("   Average loss: {:.4}", avg_loss.clone().into_scalar());

//     avg_loss
// }

// // Training function
// fn train_model(dataset_path: &str) {
//     println!("\n🚀 Starting Word Whisper Training");

//     let device = Default::default();
//     // For GPU training, uncomment:
//     // burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
//     //     &device,
//     //     Default::default(),
//     // );

//     // Build vocabulary
//     let (vocab, word_to_idx) = build_vocabulary(dataset_path);

//     // Save vocabulary for later use
//     let vocab_json = serde_json::to_string(&vocab).unwrap();
//     fs::write("vocabulary.json", vocab_json).expect("Failed to save vocabulary");
//     println!("💾 Saved vocabulary to vocabulary.json");

//     let config = WordWhisperConfig {
//         vocab_size: vocab.len(),
//         feature_dim: N_MELS,
//         hidden_dim: HIDDEN_DIM,
//         max_words: MAX_WORDS_PER_CLIP,
//     };

//     let mut model = WordWhisper::<TrainBackend>::new(&config, &device);
//     let mut optimizer = AdamConfig::new().init();

//     // Load dataset
//     let mut all_data = load_dataset(dataset_path, &word_to_idx);

//     if all_data.is_empty() {
//         panic!("❌ No data loaded! Check your dataset path and format.");
//     }

//     // Shuffle and split
//     let mut rng = thread_rng();
//     all_data.shuffle(&mut rng);

//     let split_idx = (all_data.len() as f32 * TRAIN_SPLIT) as usize;
//     let (train_data, val_data) = all_data.split_at(split_idx);

//     println!("\n🎯 Training Configuration:");
//     println!("   Train samples: {}", train_data.len());
//     println!("   Val samples: {}", val_data.len());
//     println!("   Vocabulary size: {}", vocab.len());
//     println!("   Learning rate: {}", LEARNING_RATE);
//     println!("   Dropout rate: {}", DROPOUT_RATE);
//     println!("   Max epochs: {}", EPOCHS);
//     println!("   Backend: {:?}", std::any::type_name::<TrainBackend>());

//     let mut best_val_loss = f32::INFINITY;
//     let mut current_lr = INITIAL_LR;
//     let mut epochs_without_improvement = 0;

//     // Training loop
//     for epoch in 0..EPOCHS {
//         println!("\n📊 Epoch {}/{}", epoch + 1, EPOCHS);

//         let mut epoch_loss = 0.0;
//         let mut epoch_correct = 0;
//         let mut epoch_total = 0;
//         let start_time = std::time::Instant::now();

//         // Shuffle training data each epoch
//         let mut train_batch: Vec<_> = train_data.to_vec();
//         train_batch.shuffle(&mut rng);

//         // Process each training sample
//         for (batch_idx, (mel_frames, word_indices)) in train_batch.iter().enumerate() {
//             // Debug first sample of first epoch
//             if epoch == 0 && batch_idx == 0 {
//                 println!("\n🔍 First training sample details:");
//                 println!(
//                     "   Mel frames shape: {} x {}",
//                     mel_frames.len(),
//                     mel_frames[0].len()
//                 );
//                 println!("   Word indices: {:?}", word_indices);
//                 // println!(
//                 //     "   Words: {}",
//                 //     word_indices
//                 //         .iter()
//                 //         .filter_map(|&idx| vocab.get(idx))
//                 //         .collect::<Vec<_>>()
//                 //         .join(" ")
//                 // );
//             }

//             // Prepare features tensor [1, time, mels]
//             let features_flat: Vec<f32> = mel_frames.iter().flatten().copied().collect();
//             let features = Tensor::<TrainBackend, 3>::from_data(
//                 TensorData::new(features_flat, [1, mel_frames.len(), N_MELS]),
//                 &device,
//             );

//             // Prepare targets - pad or truncate to MAX_WORDS_PER_CLIP
//             let mut padded_targets = vec![PAD_TOKEN; MAX_WORDS_PER_CLIP];
//             for (i, &idx) in word_indices.iter().enumerate().take(MAX_WORDS_PER_CLIP) {
//                 padded_targets[i] = idx;
//             }

//             let targets = Tensor::<TrainBackend, 2, Int>::from_data(
//                 TensorData::new(
//                     padded_targets.iter().map(|&x| x as i64).collect::<Vec<_>>(),
//                     [1, MAX_WORDS_PER_CLIP],
//                 ),
//                 &device,
//             );

//             // Forward pass
//             let logits = model.forward(features);

//             // Calculate accuracy (for monitoring)
//             let predictions = logits.clone().argmax(2);
//             let pred_data = predictions.into_data();
//             let pred_indices = pred_data.as_slice::<i64>().unwrap();

//             // Count correct predictions - ONLY for actual words, not padding!
//             let actual_length = word_indices.len().min(MAX_WORDS_PER_CLIP);
//             for i in 0..actual_length {
//                 epoch_total += 1;
//                 if pred_indices[i] == padded_targets[i] as i64 {
//                     epoch_correct += 1;
//                 }
//             }

//             // Debug predictions for first few batches
//             // if batch_idx < 3 && epoch % 10 == 0 {
//             println!("\n   Sample {} predictions:", batch_idx + 1);
//             println!("     Target: {:?}", &padded_targets[..actual_length]);
//             println!("     Predicted: {:?}", &pred_indices[..actual_length]);
//             // }

//             // Compute loss with padding mask and actual lengths
//             let actual_lengths = vec![word_indices.len()];
//             let loss =
//                 compute_loss_with_mask(logits, targets, PAD_TOKEN as i64, actual_lengths, &device);

//             let loss_value = loss.clone().into_scalar();
//             epoch_loss += loss_value;

//             // Backward pass
//             let grads = loss.backward();

//             // Check gradient magnitudes (debug)
//             if batch_idx == 0 && epoch % 10 == 0 {
//                 println!("\n   🔍 Gradient check:");
//                 // Note: Actual gradient inspection would require accessing internal tensors
//                 println!("     Loss value: {:.6}", loss_value);
//             }

//             let grads = GradientsParams::from_grads(grads, &model);
//             model = optimizer.step(LEARNING_RATE, model, grads);

//             // Progress indicator
//             if batch_idx % 50 == 0 && batch_idx > 0 {
//                 let batch_accuracy = (epoch_correct as f32 / epoch_total.max(1) as f32) * 100.0;
//                 print!(
//                     "\r   Progress: {}/{} batches, Loss: {:.4}, Acc: {:.1}%",
//                     batch_idx,
//                     train_batch.len(),
//                     epoch_loss / (batch_idx + 1) as f32,
//                     batch_accuracy
//                 );
//                 std::io::stdout().flush().unwrap();
//             }
//         }

//         let avg_loss = epoch_loss / train_data.len() as f32;
//         let accuracy = (epoch_correct as f32 / epoch_total.max(1) as f32) * 100.0;

//         println!(
//             "\n   Training - Loss: {:.4}, Accuracy: {:.1}%",
//             avg_loss, accuracy
//         );

//         // Validation every 5 epochs
//         if epoch % 5 == 4 || epoch == 0 {
//             println!("\n   🔍 Running validation...");

//             // Save model temporarily for validation
//             let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
//             model.clone().save_file("model_temp", &recorder).unwrap();

//             // Load model for inference
//             let inference_device: <InferenceBackend as Backend>::Device = Default::default();
//             let val_model = WordWhisper::<InferenceBackend>::new(&config, &inference_device)
//                 .load_file("model_temp", &recorder, &inference_device)
//                 .unwrap();

//             let mut val_loss = 0.0;
//             let mut val_correct = 0;
//             let mut val_total = 0;
//             let mut sample_predictions = Vec::new();

//             // Validate on subset to save time
//             let val_subset_size = val_data.len().min(50);
//             for (val_idx, (mel_frames, word_indices)) in
//                 val_data.iter().take(val_subset_size).enumerate()
//             {
//                 let features_flat: Vec<f32> = mel_frames.iter().flatten().copied().collect();
//                 let features = Tensor::<InferenceBackend, 3>::from_data(
//                     TensorData::new(features_flat, [1, mel_frames.len(), N_MELS]),
//                     &inference_device,
//                 );

//                 let mut padded_targets = vec![PAD_TOKEN; MAX_WORDS_PER_CLIP];
//                 for (i, &idx) in word_indices.iter().enumerate().take(MAX_WORDS_PER_CLIP) {
//                     padded_targets[i] = idx;
//                 }

//                 let targets = Tensor::<InferenceBackend, 2, Int>::from_data(
//                     TensorData::new(
//                         padded_targets.iter().map(|&x| x as i64).collect::<Vec<_>>(),
//                         [1, MAX_WORDS_PER_CLIP],
//                     ),
//                     &inference_device,
//                 );

//                 let logits = val_model.forward(features);

//                 // Store sample predictions for analysis
//                 if val_idx < 3 {
//                     let decoded = decode_predictions(logits.clone(), &vocab);
//                     let target_words: Vec<String> = word_indices
//                         .iter()
//                         .filter_map(|&idx| vocab.get(idx).cloned())
//                         .collect();
//                     sample_predictions.push((target_words.join(" "), decoded));
//                 }

//                 let predictions = logits.clone().argmax(2);
//                 let pred_data = predictions.into_data();
//                 let pred_indices = pred_data.as_slice::<i64>().unwrap();

//                 // Only check up to the actual number of words
//                 let actual_length = word_indices.len().min(MAX_WORDS_PER_CLIP);
//                 for i in 0..actual_length {
//                     val_total += 1;
//                     if pred_indices[i] == padded_targets[i] as i64 {
//                         val_correct += 1;
//                     }
//                 }

//                 let loss = compute_loss_with_mask(
//                     logits,
//                     targets,
//                     PAD_TOKEN as i64,
//                     vec![word_indices.len()],
//                     &inference_device,
//                 );
//                 val_loss += loss.into_scalar();
//             }

//             let avg_val_loss = val_loss / val_subset_size as f32;
//             let val_accuracy = (val_correct as f32 / val_total.max(1) as f32) * 100.0;

//             println!(
//                 "   Validation - Loss: {:.4}, Accuracy: {:.1}%",
//                 avg_val_loss, val_accuracy
//             );

//             // Show sample predictions
//             println!("\n   📝 Sample predictions:");
//             for (i, (target, pred)) in sample_predictions.iter().enumerate() {
//                 println!("     {}. Target: '{}'", i + 1, target);
//                 println!("        Predicted: '{}'", pred);
//             }

//             // Save best model
//             if avg_val_loss < best_val_loss {
//                 best_val_loss = avg_val_loss;
//                 epochs_without_improvement = 0;
//                 model.clone().save_file("model_best", &recorder).unwrap();
//                 println!("\n   💾 Saved new best model (loss: {:.4})", best_val_loss);
//             } else {
//                 epochs_without_improvement += 1;
//             }

//             let _ = fs::remove_file("model_temp.mpk");
//         }

//         let epoch_time = start_time.elapsed().as_secs_f32();
//         println!("   Epoch time: {:.1}s", epoch_time);
//         println!("   Learning rate: {:.6}", current_lr);

//         // Early stopping conditions
//         if accuracy > 90.0 {
//             println!("\n✨ Early stopping - high accuracy achieved!");
//             break;
//         }

//         if epochs_without_improvement > 15 {
//             println!("\n⚠️  Early stopping - no improvement for 15 epochs");
//             break;
//         }

//         // Learning rate decay
//         current_lr = (current_lr * LR_DECAY_RATE).max(MIN_LR);
//     }

//     // Save final model
//     println!("\n💾 Saving final model...");
//     let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
//     model
//         .save_file("model", &recorder)
//         .expect("Failed to save model");

//     println!("\n✅ Training complete!");
//     println!("   Best validation loss: {:.4}", best_val_loss);
// }

// // Real-time transcription
// fn transcribe_realtime() {
//     println!("🎤 Loading model for transcription...");

//     let device = Default::default();

//     // Load vocabulary
//     let vocab_json = fs::read_to_string("vocabulary.json")
//         .expect("Vocabulary file not found. Train the model first!");
//     let vocab: Vec<String> = serde_json::from_str(&vocab_json).unwrap();
//     println!("✅ Loaded vocabulary with {} words", vocab.len());

//     let config = WordWhisperConfig {
//         vocab_size: vocab.len(),
//         feature_dim: N_MELS,
//         hidden_dim: HIDDEN_DIM,
//         max_words: MAX_WORDS_PER_CLIP,
//     };

//     let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
//     let model = WordWhisper::<InferenceBackend>::new(&config, &device)
//         .load_file("model_best", &recorder, &device)
//         .or_else(|_| {
//             println!("⚠️  Best model not found, trying final model...");
//             WordWhisper::<InferenceBackend>::new(&config, &device)
//                 .load_file("model", &recorder, &device)
//         })
//         .expect("No model found. Train first!");

//     println!("✅ Model loaded successfully");

//     // Audio buffer for real-time capture
//     let audio_buffer = Arc::new(Mutex::new(VecDeque::<f32>::new()));
//     let audio_buffer_clone = Arc::clone(&audio_buffer);

//     // Setup audio input
//     let host = cpal::default_host();
//     let input_device = host.default_input_device().expect("No input device");
//     println!(
//         "📢 Using audio device: {}",
//         input_device.name().unwrap_or("Unknown".to_string())
//     );

//     let input_config = input_device
//         .default_input_config()
//         .expect("Failed to get config");
//     let channels = input_config.channels();
//     println!("   Channels: {}", channels);
//     println!("   Sample rate: {} Hz", input_config.sample_rate().0);

//     let stream = match input_config.sample_format() {
//         cpal::SampleFormat::F32 => {
//             input_device
//                 .build_input_stream(
//                     &input_config.into(),
//                     move |data: &[f32], _: &_| {
//                         // Convert to mono if needed
//                         let mono: Vec<f32> = if channels > 1 {
//                             data.chunks(channels as usize)
//                                 .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
//                                 .collect()
//                         } else {
//                             data.to_vec()
//                         };

//                         let mut buffer = audio_buffer_clone.lock().unwrap();
//                         buffer.extend(mono.iter());

//                         // Keep only last 15 seconds
//                         while buffer.len() > SAMPLE_RATE as usize * 15 {
//                             buffer.pop_front();
//                         }
//                     },
//                     |err| eprintln!("Stream error: {}", err),
//                     None,
//                 )
//                 .expect("Failed to build stream")
//         }
//         _ => panic!("Unsupported audio format"),
//     };

//     stream.play().expect("Failed to start stream");
//     println!("\n🎯 Listening... (speak clearly in complete sentences)\n");
//     println!("Press Ctrl+C to stop\n");

//     let mut last_transcription = String::new();
//     let mut silence_counter = 0;

//     loop {
//         std::thread::sleep(std::time::Duration::from_millis(1500)); // Process every 1.5 seconds

//         let buffer = audio_buffer.lock().unwrap();
//         if buffer.len() >= SAMPLE_RATE as usize * 2 {
//             // At least 2 seconds
//             let audio: Vec<f32> = buffer.iter().copied().collect();
//             drop(buffer);

//             // Calculate RMS (root mean square) for voice activity detection
//             let rms = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();

//             if rms > 0.01 {
//                 // Voice detected
//                 silence_counter = 0;
//                 println!("🔊 Processing audio (RMS: {:.3})...", rms);

//                 let mel_frames = compute_mel_spectrogram(&audio);

//                 // Normalize energy (important for matching training data distribution)
//                 let avg_energy: f32 = mel_frames
//                     .iter()
//                     .flat_map(|frame| frame.iter())
//                     .map(|&x| x.abs())
//                     .sum::<f32>()
//                     / (mel_frames.len() * N_MELS) as f32;

//                 let target_energy = 2.0; // Typical training data energy
//                 let scale_factor = target_energy / (avg_energy + 1e-6);

//                 let normalized_frames: Vec<Vec<f32>> = mel_frames
//                     .into_iter()
//                     .map(|frame| frame.into_iter().map(|x| x * scale_factor).collect())
//                     .collect();

//                 if normalized_frames.len() > 50 {
//                     // Enough frames for processing
//                     let features_flat: Vec<f32> =
//                         normalized_frames.iter().flatten().copied().collect();
//                     let features = Tensor::<InferenceBackend, 3>::from_data(
//                         TensorData::new(features_flat, [1, normalized_frames.len(), N_MELS]),
//                         &device,
//                     );

//                     println!("   Feature shape: {:?}", features.dims());

//                     let predictions = model.forward(features);
//                     let text = decode_predictions(predictions, &vocab);

//                     if !text.is_empty() && text != last_transcription {
//                         println!("\n✨ Transcription: {}\n", text);
//                         last_transcription = text;
//                     } else if text.is_empty() {
//                         println!("   (No words detected)");
//                     }
//                 }
//             } else {
//                 silence_counter += 1;
//                 if silence_counter % 4 == 0 {
//                     // Every 6 seconds of silence
//                     println!("🔇 Waiting for speech...");
//                 }
//             }
//         }
//     }
// }

// fn main() {
//     let args: Vec<String> = std::env::args().collect();

//     println!("\n🎙️  Word Whisper - Word-level Speech Recognition");

//     match args.get(1).map(|s| s.as_str()) {
//         Some("train") => {
//             if let Some(dataset_path) = args.get(2) {
//                 if !Path::new(dataset_path).exists() {
//                     eprintln!("❌ Error: Dataset path '{}' does not exist!", dataset_path);
//                     std::process::exit(1);
//                 }
//                 train_model(dataset_path);
//             } else {
//                 eprintln!("❌ Error: Missing dataset path!");
//                 eprintln!("Usage: cargo run -- train <dataset_path>");
//                 eprintln!("\nExample: cargo run -- train ./my_audio_dataset");
//             }
//         }
//         Some("transcribe") => {
//             if !Path::new("model.mpk").exists() && !Path::new("model_best.mpk").exists() {
//                 eprintln!("❌ Error: No trained model found!");
//                 eprintln!("Please train a model first using: cargo run -- train <dataset_path>");
//                 std::process::exit(1);
//             }
//             transcribe_realtime();
//         }
//         _ => {
//             println!("Usage:");
//             println!("  cargo run -- train <dataset_path>    Train the model");
//             println!("  cargo run -- transcribe              Real-time transcription");
//             println!("\nDataset format:");
//             println!("  - WAV files: 16kHz, mono");
//             println!("  - TXT files: corresponding transcripts");
//             println!("\nExample:");
//             println!("  dataset/");
//             println!("    sample1.wav");
//             println!("    sample1.txt (contains: 'hello world')");
//             println!("    sample2.wav");
//             println!("    sample2.txt");
//         }
//     }
// }
