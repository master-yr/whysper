#![recursion_limit = "256"]

use burn::backend::wgpu::graphics::OpenGl;
use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::module::AutodiffModule;
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
use burn_cuda::Cuda;
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
#[cfg(not(feature = "cuda"))]
type TrainBackend = Autodiff<NdArray>;
#[cfg(not(feature = "cuda"))]
type InferenceBackend = NdArray;

#[cfg(feature = "cuda")]
type TrainBackend = Autodiff<Cuda>;
#[cfg(feature = "cuda")]
type InferenceBackend = Cuda;

// type TrainBackend = Autodiff<Wgpu>;
// type InferenceBackend = Wgpu;

// type TrainBackend = Autodiff<burn::backend::Candle>;
// type InferenceBackend = Candle;

// Audio constants
const SAMPLE_RATE: u32 = 16000; // 16kHz for speech
const FRAME_SIZE: usize = 512; // FFT window size
const HOP_SIZE: usize = 160; // Step between frames
const N_MELS: usize = 64; // Mel frequency bins

// Model constants
const MAX_SAMPLES_PER_WORD: usize = 4000;
const HIDDEN_DIM: usize = 256; // Hidden layer size
const DROPOUT_RATE: f64 = 0.4; // Less aggressive dropout

// Training constants
const LEARNING_RATE: f64 = 0.05;
const LEARNING_RATE_DECAY: f64 = 0.75; // Decay factor per epoch
const EPOCHS: usize = 10;
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
    println!("   Initial learning rate: {}", LEARNING_RATE);
    println!("   Learning rate decay: {}", LEARNING_RATE_DECAY);

    let mut best_val_acc = 0.0;
    let training_start = std::time::Instant::now();
    let mut epoch_times = Vec::new();

    // Training loop
    for epoch in 0..EPOCHS {
        let epoch_start = std::time::Instant::now();

        // Calculate decayed learning rate
        let current_lr = LEARNING_RATE * LEARNING_RATE_DECAY.powi(epoch as i32);

        println!(
            "\n📊 Epoch {}/{} (LR: {:.6})",
            epoch + 1,
            EPOCHS,
            current_lr
        );

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
            #[cfg(feature = "cuda")]
            let pred_idx = pred.into_data().as_slice::<i32>().unwrap()[0] as usize;
            #[cfg(not(feature = "cuda"))]
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
                    "\r   [{:>3.0}%] Loss: {:.4}, Acc: {:.1}% | '{}' → '{}' {}",
                    progress,
                    avg_loss,
                    current_acc,
                    actual_word,
                    predicted_word,
                    if pred_idx == *label { "✓" } else { "✗" }
                );
                std::io::stdout().flush().unwrap();
            }

            // Backward pass with decayed learning rate
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(current_lr, model, grads);
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

        // Fixed validation logic
        if epoch % 5 == 4 || epoch == 0 {
            println!("   🔍 Validating...");

            let val_model = model.clone().valid();
            // Save current model
            // let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
            // model.clone().save_file("model_temp", &recorder).unwrap();

            // Create inference model and load weights
            let inference_device: <InferenceBackend as Backend>::Device = Default::default();

            // // CRITICAL: Must create model with SAME config and properly load
            // let mut val_model = WordRecognizer::<InferenceBackend>::new(&config, &inference_device);

            // Load the saved weights - this was the issue!
            // val_model = val_model
            //     .load_file("model_temp", &recorder, &inference_device)
            //     .expect("Failed to load model for validation");

            // Set to evaluation mode (disables dropout)
            // val_model = val_model.valid();

            let mut val_correct = 0;
            let mut val_loss = 0.0;
            let mut sample_results = Vec::new();

            for (idx, (mel_frames, label)) in val_data.iter().enumerate() {
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
                #[cfg(feature = "cuda")]
                let pred_idx = pred.into_data().as_slice::<i32>().unwrap()[0] as usize;
                #[cfg(not(feature = "cuda"))]
                let pred_idx = pred.into_data().as_slice::<i64>().unwrap()[0] as usize;

                // Debug: show first few predictions
                if idx < 5 {
                    sample_results.push((
                        vocab[*label].clone(),
                        vocab[pred_idx].clone(),
                        pred_idx == *label,
                    ));
                }

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

            // Show sample predictions for debugging
            println!("   Sample predictions:");
            for (actual, predicted, correct) in sample_results {
                println!(
                    "      '{}' → '{}' {}",
                    actual,
                    predicted,
                    if correct { "✓" } else { "✗" }
                );
            }

            // Save best model (fix the print statement)
            if val_acc > best_val_acc {
                let prev_best = best_val_acc; // Store previous before updating
                best_val_acc = val_acc;
                // model.clone().save_file("model_best", &recorder).unwrap();
                println!(
                    "   💾 New best model saved! ({:.1}% → {:.1}%)",
                    prev_best, best_val_acc
                );
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
// fn recognize_realtime() {
//     println!("🎤 Loading model for recognition...");

//     let device = Default::default();

//     // Load vocabulary
//     let vocab_json =
//         fs::read_to_string("vocabulary.json").expect("Vocabulary not found. Train first!");
//     let vocab: Vec<String> = serde_json::from_str(&vocab_json).unwrap();

//     let config = WordRecognizerConfig {
//         vocab_size: vocab.len(),
//         feature_dim: N_MELS,
//         hidden_dim: HIDDEN_DIM,
//     };

//     let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();
//     let model = WordRecognizer::<InferenceBackend>::new(&config, &device)
//         .load_file("model_best", &recorder, &device)
//         .or_else(|_| {
//             WordRecognizer::<InferenceBackend>::new(&config, &device)
//                 .load_file("model", &recorder, &device)
//         })
//         .expect("No model found!");

//     println!("✅ Model loaded - recognizing {} words:", vocab.len());
//     for (i, word) in vocab.iter().enumerate() {
//         print!("{:>15}", word);
//         if (i + 1) % 5 == 0 {
//             println!();
//         }
//     }
//     println!("\n");

//     // Audio buffer
//     let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
//     let audio_buffer_clone = Arc::clone(&audio_buffer);
//     let is_recording = Arc::new(Mutex::new(false));
//     let is_recording_clone = Arc::clone(&is_recording);

//     // Setup audio
//     let host = cpal::default_host();
//     let input_device = host.default_input_device().expect("No input device");
//     let input_config = input_device
//         .default_input_config()
//         .expect("Failed to get config");
//     let channels = input_config.channels();

//     let stream = match input_config.sample_format() {
//         cpal::SampleFormat::F32 => input_device
//             .build_input_stream(
//                 &input_config.into(),
//                 move |data: &[f32], _: &_| {
//                     if *is_recording_clone.lock().unwrap() {
//                         let mono: Vec<f32> = if channels > 1 {
//                             data.chunks(channels as usize)
//                                 .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
//                                 .collect()
//                         } else {
//                             data.to_vec()
//                         };

//                         audio_buffer_clone.lock().unwrap().extend(mono);
//                     }
//                 },
//                 |err| eprintln!("Stream error: {}", err),
//                 None,
//             )
//             .expect("Failed to build stream"),
//         _ => panic!("Unsupported format"),
//     };

//     stream.play().expect("Failed to start stream");

//     println!("🎯 Press ENTER to record a word, Ctrl+C to quit\n");

//     loop {
//         // Wait for enter
//         let mut input = String::new();
//         std::io::stdin().read_line(&mut input).unwrap();

//         // Clear buffer and start recording
//         audio_buffer.lock().unwrap().clear();
//         *is_recording.lock().unwrap() = true;

//         println!("🔴 Recording... (say one word clearly)");
//         std::thread::sleep(std::time::Duration::from_millis(1500));

//         // Stop recording
//         *is_recording.lock().unwrap() = false;
//         let audio = audio_buffer.lock().unwrap().clone();

//         if audio.len() > SAMPLE_RATE as usize / 4 {
//             // At least 0.25 seconds
//             println!("🔊 Processing {} samples...", audio.len());

//             let mel_frames = compute_mel_spectrogram(&audio);

//             if mel_frames.len() > 10 {
//                 let features_flat: Vec<f32> = mel_frames.iter().flatten().copied().collect();
//                 let features = Tensor::<InferenceBackend, 3>::from_data(
//                     TensorData::new(features_flat, [1, mel_frames.len(), N_MELS]),
//                     &device,
//                 );

//                 let logits = model.forward(features);
//                 let probs = softmax(logits, 1);
//                 let pred = probs.clone().argmax(1);

//                 let pred_idx = pred.into_data().as_slice::<i64>().unwrap()[0] as usize;
//                 let confidence = probs.into_data().as_slice::<f32>().unwrap()[pred_idx];

//                 println!(
//                     "\n✨ Recognized: '{}' (confidence: {:.1}%)\n",
//                     vocab[pred_idx],
//                     confidence * 100.0
//                 );
//             } else {
//                 println!("❌ Too short, try again\n");
//             }
//         } else {
//             println!("❌ No audio captured\n");
//         }
//     }
// }

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
    let actual_sample_rate = input_config.sample_rate().0;

    // Resample ratio if needed
    let resample_ratio = SAMPLE_RATE as f32 / actual_sample_rate as f32;
    println!(
        "📊 Audio config: {}Hz, {} channels",
        actual_sample_rate, channels
    );
    if actual_sample_rate != SAMPLE_RATE {
        println!(
            "   Will resample from {}Hz to {}Hz",
            actual_sample_rate, SAMPLE_RATE
        );
    }

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

                        // Simple resampling if needed
                        let resampled = if actual_sample_rate != SAMPLE_RATE {
                            let mut output = Vec::new();
                            let mono_len = mono.len();
                            for i in 0..(mono_len as f32 * resample_ratio) as usize {
                                let src_idx = (i as f32 / resample_ratio) as usize;
                                if src_idx < mono_len {
                                    output.push(mono[src_idx]);
                                }
                            }
                            output
                        } else {
                            mono
                        };

                        audio_buffer_clone.lock().unwrap().extend(resampled);
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

        // Record with visual feedback
        let mut max_amplitude = 0.0f32;
        for i in 0..30 {
            // 3 seconds max, checking every 100ms
            std::thread::sleep(std::time::Duration::from_millis(100));

            let current_audio = audio_buffer.lock().unwrap().clone();
            if !current_audio.is_empty() {
                let current_max = current_audio
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0f32, f32::max);
                max_amplitude = max_amplitude.max(current_max);

                // Visual level meter
                let level = (current_max * 50.0).min(50.0) as usize;
                print!("\r🔴 Recording [");
                for j in 0..50 {
                    if j < level {
                        print!("█");
                    } else {
                        print!("░");
                    }
                }
                print!("] {:.1}s", (i + 1) as f32 * 0.1);
                std::io::stdout().flush().unwrap();
            }

            // Stop if we have enough audio with voice activity
            if current_audio.len() > SAMPLE_RATE as usize && max_amplitude > 0.01 {
                break;
            }
        }
        println!(); // New line after recording

        // Stop recording
        *is_recording.lock().unwrap() = false;
        let mut audio = audio_buffer.lock().unwrap().clone();

        if audio.len() > SAMPLE_RATE as usize / 4 && max_amplitude > 0.005 {
            println!(
                "🔊 Processing {} samples (max amplitude: {:.3})...",
                audio.len(),
                max_amplitude
            );

            // Normalize audio
            let max_val = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            if max_val > 0.0 {
                let scale = 0.95 / max_val; // Normalize to 95% of maximum
                for sample in audio.iter_mut() {
                    *sample *= scale;
                }
            }

            // Apply pre-emphasis filter (boost high frequencies)
            let pre_emphasis = 0.97;
            for i in (1..audio.len()).rev() {
                audio[i] -= pre_emphasis * audio[i - 1];
            }

            // Trim silence from beginning and end
            let energy_threshold = 0.01;
            let window_size = (0.02 * SAMPLE_RATE as f32) as usize; // 20ms windows

            // Find start of speech
            let mut start_idx = 0;
            for i in (0..audio.len()).step_by(window_size) {
                let window_end = (i + window_size).min(audio.len());
                let energy: f32 = audio[i..window_end].iter().map(|&x| x * x).sum();
                let avg_energy = energy / window_size as f32;
                if avg_energy > energy_threshold {
                    start_idx = i.saturating_sub(window_size); // Include one window before
                    break;
                }
            }

            // Find end of speech
            let mut end_idx = audio.len();
            for i in (0..audio.len()).step_by(window_size).rev() {
                let window_end = (i + window_size).min(audio.len());
                let energy: f32 = audio[i..window_end].iter().map(|&x| x * x).sum();
                let avg_energy = energy / window_size as f32;
                if avg_energy > energy_threshold {
                    end_idx = window_end.min(audio.len());
                    break;
                }
            }

            // Extract speech segment
            if start_idx < end_idx {
                audio = audio[start_idx..end_idx].to_vec();
                println!(
                    "   Detected speech: {:.2}s to {:.2}s",
                    start_idx as f32 / SAMPLE_RATE as f32,
                    end_idx as f32 / SAMPLE_RATE as f32
                );
            }

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

                #[cfg(feature = "cuda")]
                let pred_idx = pred.into_data().as_slice::<i32>().unwrap()[0] as usize;
                #[cfg(not(feature = "cuda"))]
                let pred_idx = pred.into_data().as_slice::<i64>().unwrap()[0] as usize;

                let confidence = probs.clone().into_data().as_slice::<f32>().unwrap()[pred_idx];

                // Show top 3 predictions
                let probs_data = probs.into_data();
                let probs_slice = probs_data.as_slice::<f32>().unwrap();
                let mut scores: Vec<(usize, f32)> = probs_slice
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i, p))
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                println!("\n✨ Top predictions:");
                for (i, (idx, prob)) in scores.iter().take(3).enumerate() {
                    let marker = if i == 0 { "→" } else { " " };
                    println!("   {} {}: {:.1}%", marker, vocab[*idx], prob * 100.0);
                }

                if confidence < 0.5 {
                    println!("\n⚠️  Low confidence - please speak more clearly\n");
                } else {
                    println!(
                        "\n✅ Recognized: '{}' (confidence: {:.1}%)\n",
                        vocab[pred_idx],
                        confidence * 100.0
                    );
                }
            } else {
                println!("❌ Audio too short after trimming, try again\n");
            }
        } else {
            if max_amplitude < 0.005 {
                println!("❌ No voice detected (too quiet), please speak louder\n");
            } else {
                println!("❌ No audio captured\n");
            }
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
