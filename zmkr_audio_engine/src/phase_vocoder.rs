use std::{
    f32::consts::PI,
    ops::DerefMut,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use ringbuf::{traits::*, HeapRb};
use symphonia::core::dsp::complex;


pub struct PhaseVocoder {
    curr_frame: usize,
    phase_acc: Vec<f32>,
    prev_synth_map: Vec<Complex<f32>>,
    windowing_func: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    fft_scratch_buf: Vec<Complex<f32>>,
    fft_buf: Vec<Complex<f32>>,
    fft_out: [Vec<Complex<f32>>; 2],
    pub output_buf: Vec<f32>,
}

impl PhaseVocoder {
    pub const WINDOW_SZ: usize = 2048;
    const ANALYSIS_HOP_SZ: usize = Self::WINDOW_SZ / 4;
    const SYNTHESIS_HOP_SZ: usize = Self::ANALYSIS_HOP_SZ;
    pub fn new() -> Self {
        let hann_window: Vec<f32> = (0..Self::WINDOW_SZ)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (Self::WINDOW_SZ as f32 - 1.0)).cos()))
            .collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(Self::WINDOW_SZ);
        let ifft = planner.plan_fft_inverse(Self::WINDOW_SZ);
        let fft_scratch_buf = vec![Complex::new(0.0f32, 0.0f32); fft.get_inplace_scratch_len()];
        let fft_buf = vec![Complex::new(0.0f32, 0.0f32); Self::WINDOW_SZ];
        let output_buf = vec![0.0f32; Self::WINDOW_SZ * 2];
        let fft_out = [
            vec![Complex::new(0.0f32, 0.0f32); Self::WINDOW_SZ],
            vec![Complex::new(0.0f32, 0.0f32); Self::WINDOW_SZ]
        ];
        
        let mut pv = PhaseVocoder {
            curr_frame: 0,
            phase_acc: Vec::<f32>::new(),
            prev_synth_map: Vec::<Complex<f32>>::new(),
            windowing_func: hann_window,
            fft,
            ifft,
            fft_scratch_buf,
            fft_buf,
            output_buf,
            fft_out,
        };

        for _ in 0..Self::WINDOW_SZ {
            let num = Complex::<f32>::new(0f32, 0f32);
            pv.phase_acc.push(0f32);
            pv.prev_synth_map.push(num);
        };

        pv
    }

    pub fn pv_analyze(&mut self, orig_data: &[f32]) {
        for channel_idx in 0..2 {
            for n in 0..Self::WINDOW_SZ {
                let sample = orig_data[n * 2 + channel_idx];
                // WARNING: Removed window for passthrough testing
                // self.fft_buf[n] = Complex::new(sample * self.windowing_func[n], 0.0);
                self.fft_buf[n] = Complex::new(sample, 0.0);
            }
        
            self.fft.process_with_scratch(&mut self.fft_buf, &mut self.fft_scratch_buf);
            
            self.fft_out[channel_idx].copy_from_slice(&self.fft_buf);
        }
    }

    pub fn pv_synthesize(&mut self) {
        self.output_buf.fill(0.0);

        for channel_idx in 0..2 {
            self.fft_buf.copy_from_slice(&self.fft_out[channel_idx]);

            self.ifft.process_with_scratch(&mut self.fft_buf, &mut self.fft_scratch_buf);

            for n in 0..Self::WINDOW_SZ {
                // WARNING: Removed windowing for passthrough test
                // let sample = (self.fft_buf[n].re / Self::WINDOW_SZ as f32) * self.windowing_func[n];
                
                let sample = self.fft_buf[n].re / Self::WINDOW_SZ as f32;
                self.output_buf[n * 2 + channel_idx] = sample;
            }
        }
    }

    pub fn pv_run(&mut self, orig_data: &[f32], tempo: i32) {
        self.pv_analyze(orig_data);
        self.pv_synthesize();
    }
}