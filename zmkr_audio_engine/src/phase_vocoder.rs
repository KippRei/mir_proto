use std::{
    f32::consts::PI,
    sync::{Arc},
};

use rustfft::{num_complex::Complex, Fft, FftPlanner};


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
    temp_output_buf: Vec<f32>,
}

impl PhaseVocoder {
    pub const WINDOW_SZ: usize = 2048;
    const ANALYSIS_HOP_SZ: usize = Self::WINDOW_SZ / 4;
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
        let temp_output_buf = vec![0.0f32; Self::WINDOW_SZ * 2];
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
            fft_out,
            output_buf,
            temp_output_buf,
        };

        for _ in 0..Self::WINDOW_SZ {
            let num = Complex::<f32>::new(0f32, 0f32);
            pv.phase_acc.push(0f32);
            pv.prev_synth_map.push(num);
        };

        pv
    }

    pub fn pv_analyze(&mut self, orig_data: &[f32], start_idx: usize) {
        for channel_idx in 0..2 {
            for n in 0..Self::WINDOW_SZ {
                let sample = orig_data[(n + start_idx) * 2 + channel_idx];
                self.fft_buf[n] = Complex::new(sample * self.windowing_func[n], 0.0);
            }
        
            self.fft.process_with_scratch(&mut self.fft_buf, &mut self.fft_scratch_buf);
            
            self.fft_out[channel_idx].copy_from_slice(&self.fft_buf);
        }
    }

    pub fn pv_synthesize(&mut self, start_idx: usize) {
        for channel_idx in 0..2 {
            self.fft_buf.copy_from_slice(&self.fft_out[channel_idx]);
            // TODO: Implement actual time scaling
            self.ifft.process_with_scratch(&mut self.fft_buf, &mut self.fft_scratch_buf);

            for n in 0..Self::WINDOW_SZ {
                let sample = (self.fft_buf[n].re / Self::WINDOW_SZ as f32) * self.windowing_func[n];
                let data_buf_idx = n + start_idx;
                // TODO: Adjust scaling to account for actual stretch factor
                if data_buf_idx < Self::WINDOW_SZ {
                    self.output_buf[data_buf_idx * 2 + channel_idx] += sample / 4f32;
                }
                else {
                    let temp_idx = data_buf_idx - Self::WINDOW_SZ;
                    // TODO: Adjust scaling to account for actual stretch factor
                    self.temp_output_buf[temp_idx * 2 + channel_idx] += sample / 4f32;
                }
            }
        }
    }

    pub fn pv_run(&mut self, orig_data: &[f32], tempo: f32) {
        self.output_buf.fill(0.0);
        for i in 0..self.temp_output_buf.len() - 1 {
            self.output_buf[i] = self.temp_output_buf[i];
        }
        self.temp_output_buf.fill(0.0);
        let num_windows = orig_data.len() / (Self::WINDOW_SZ * 2);
        let stretch_factor = tempo / 124f32;
        let synth_hop = Self::ANALYSIS_HOP_SZ * stretch_factor as usize;

        for i in 0..num_windows {
            let start_idx = i * Self::ANALYSIS_HOP_SZ;
            self.pv_analyze(orig_data, start_idx);

            let synth_hop_start = i * synth_hop;
            self.pv_synthesize(synth_hop_start);
        }
    }
}