const SAMPLE_RATE: u32 = 16000; // 16kHz for speech
const MAX_SAMPLES_PER_WORD: usize = 5;
const FRAME_SIZE: usize = 512; // FFT window size
const HOP_SIZE: usize = 160; // Step between frames
const N_MELS: usize = 80; // Mel frequency bins

WS | HDIM |MAX_S|  LR |EPC |SPLIT| DRP || ECH-ACC | T-ACC
5  | 258  | 100 | .005| 50 | 0.8 | 0.3 || 75%     | 26% 
2  | 258  | 400 | .005| 50 | 0.8 | 0.3 ||      |  
