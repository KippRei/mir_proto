use std::{
    f64::consts::PI,
    ops::DerefMut,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use ndarray::{s, Array, Array2, Array3};
use rustfft::{num_complex::Complex, Fft, FftPlanner};

use crate::audio_player::_Channel;

pub struct PhaseVocoder {
    curr_frame: usize,
    prev_phase_map: Vec<f64>,
    prev_synth_map: Vec<Complex<f64>>,
}

impl PhaseVocoder {
    const WINDOW_SZ: usize = 1024;
    const ANALYSIS_HOP_SZ: usize = Self::WINDOW_SZ / 4;
    const SYNTHESIS_HOP_SZ: usize = Self::ANALYSIS_HOP_SZ;
    pub fn new() -> Self {
        let mut pv = PhaseVocoder {
            curr_frame: 0,
            prev_phase_map: Vec::<f64>::new(),
            prev_synth_map: Vec::<Complex<f64>>::new(),
        };

        for _ in 0..Self::WINDOW_SZ {
            let num = Complex::<f64>::new(0f64, 0f64);
            pv.prev_phase_map.push(0f64);
            pv.prev_synth_map.push(num);
        };

        pv
    }

    pub fn start_pv(&mut self, channel_map: &Arc<Mutex<Vec<_Channel>>>) {
        let channel_map_arc_clone = channel_map.clone();
        let mut planner: FftPlanner<f64> = FftPlanner::new();
        let ifft: Arc<dyn Fft<f64>> = planner.plan_fft_inverse(Self::WINDOW_SZ);

        // let num_frames = spectral_data.shape()[0];
        // let num_channels = 2;
        // let output_len = (num_frames - 1) * Self::SYNTHESIS_HOP_SZ + Self::WINDOW_SZ;

        let window: Vec<f64> = (0..Self::WINDOW_SZ)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f64 / (Self::WINDOW_SZ as f64 - 1.0)).cos()))
            .collect();


        // TODO: Implement loop for pv
        // loop {
        //     let mut channel_map_lock = channel_map_arc_clone.lock().unwrap();
        //     let channel_map = channel_map_lock.deref_mut();
        //     for channel in channel_map {
        //         if channel.is_playing() {
        //             let raw_data = channel.data.as_ref().unwrap();
        //             let mut buffer_lock = channel.data.as_ref().lock().unwrap();
        //             let mut buffer = buffer_lock.deref_mut();
        //             self.pv_synthesize_frame(&raw_data, channel.curr_frame, &mut buffer, &ifft, &window);
        //             channel.curr_frame += 1;
        //         }
        //     }

            // let channel_map = channel_map_lock.deref_mut();

            // for j in channel_map {
            //     if j.is_loaded() {
            //         let mut buffer_lock = j.data.as_ref().lock().unwrap();
            //         let buffer = buffer_lock.deref_mut();
            //         for k in buffer {
            //             println!("{}", k);
            //         }
            //     }
            // }

            // std::mem::drop(channel_map_lock);
            // thread::sleep(Duration::from_millis(30));
        //}
    }

    pub fn pv_analyze(&self, orig_data: &Array2<f64>) -> Array3<Complex<f64>> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(Self::WINDOW_SZ);

        // Get number of samples
        let num_samples = orig_data.shape()[0];

        // Calculate number of frames for our output array
        let num_frames = ((num_samples - Self::WINDOW_SZ) / Self::ANALYSIS_HOP_SZ) + 1;

        // Will hold spectral data after STFTs (reshaped before returning with .into_shape((num_frames, Self::WINDOW_SZ, 2)))
        let mut spectral_frames = vec![Complex::new(0.0, 0.0); num_frames * Self::WINDOW_SZ * 2];

        // Holds starting index for current frame
        let mut frame_start = 0;
        // Holds multiplier to get to get correct index for spectral_frames
        let mut frame_idx_mult = 0;

        // The Hann Window vector for windowing frames
        let window: Vec<f64> = (0..Self::WINDOW_SZ)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f64 / (Self::WINDOW_SZ as f64 - 1.0)).cos()))
            .collect();

        for channel_idx in 0..2 {
            while frame_start + Self::WINDOW_SZ <= num_samples {
                let mut buffer: Vec<Complex<f64>> = (0..Self::WINDOW_SZ)
                    .map(|n| {
                        // Get the original sample
                        let sample_value = orig_data[[frame_start + n, channel_idx]];
                        // Apply windowing function
                        let windowed_sample = sample_value * window[n];
                        // Converts data to format FFT needs (real + imaginary)
                        Complex::new(windowed_sample, 0.0)
                    })
                    .collect();

                // Perform FFT on buffer
                fft.process(&mut buffer);

                // Copy result into spectral_frames vector
                for i in 0..Self::WINDOW_SZ {
                    spectral_frames[(frame_idx_mult * Self::WINDOW_SZ) + i] = buffer[i];
                }

                frame_start += Self::ANALYSIS_HOP_SZ;
                frame_idx_mult += 1;
            }
            frame_start = 0;
        }

        // Convert vector into ndarray so we can change shape for return
        let spectral_frames_arr = Array::from_vec(spectral_frames);
        // println!("spectral_frames_count:{:?}", spectral_frames_arr.shape());
        // println!("frames:{:?},window_sz:{:?}", num_frames, Self::WINDOW_SZ);
        // println!("new arr count:{}", num_frames*Self::WINDOW_SZ*2);

        // Take ownership and change shape of spectral frames array and return
        spectral_frames_arr
            .to_shape((num_frames, Self::WINDOW_SZ, 2))
            .unwrap()
            .into_owned()
    }

    pub fn pv_synthesize_full(&self, spectral_data: &Array3<Complex<f64>>) -> Array2<f64> {
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(Self::WINDOW_SZ);

        let num_frames = spectral_data.shape()[0];
        let num_channels = 2;

        let output_len = (num_frames - 1) * Self::SYNTHESIS_HOP_SZ + Self::WINDOW_SZ;

        let mut output_data = Array2::<f64>::zeros((output_len, num_channels));

        let window: Vec<f64> = (0..Self::WINDOW_SZ)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f64 / (Self::WINDOW_SZ as f64 - 1.0)).cos()))
            .collect();

        for frame_idx in 0..num_frames {
            let analysis_frame = spectral_data.slice(s![frame_idx, .., ..]);
            let synth_start_idx = frame_idx * Self::SYNTHESIS_HOP_SZ;

            for channel_idx in 0..2 {
                let mut ifft_buffer = analysis_frame.slice(s![.., channel_idx]).to_vec();
                ifft.process(&mut ifft_buffer);

                for (i, val) in ifft_buffer.iter().enumerate() {
                    let real_sample = val.re / Self::WINDOW_SZ as f64;
                    let windowed_sample = real_sample * window[i];

                    let out_idx = synth_start_idx + i;

                    output_data[[out_idx, channel_idx]] += windowed_sample;
                }
            }
        }

        output_data
    }

    pub fn pv_synthesize_frame(
        &mut self,
        spectral_data: &Array3<Complex<f64>>,
        curr_frame_idx: usize,
        output_buffer: &mut Array2<f64>,
        ifft: &Arc<dyn Fft<f64>>,
        window: &Vec<f64>,
    ) {
        let analysis_frame = spectral_data.slice(s![curr_frame_idx, .., ..]);
        let synth_start_idx = curr_frame_idx * Self::SYNTHESIS_HOP_SZ;

        for channel_idx in 0..2 {
            let mut ifft_buffer = Self::process_frame(self, analysis_frame.slice(s![.., channel_idx]).to_vec());
            ifft.process(&mut ifft_buffer);

            for (i, val) in ifft_buffer.iter().enumerate() {
                let real_sample = (val.re / Self::WINDOW_SZ as f64) * (Self::SYNTHESIS_HOP_SZ as f64/ Self::WINDOW_SZ as f64);
                let windowed_sample = real_sample * window[i];

                let out_idx = synth_start_idx + i;
                // if out_idx < 10 {
                //     println!("{},{}",curr_frame_idx, windowed_sample);
                // }
                output_buffer[[out_idx, channel_idx]] += windowed_sample;
            }
        }
    }

    fn process_frame(&mut self, frame: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        let mut processed_frame = Vec::<Complex<f64>>::new();
        for (i, val) in frame.iter().enumerate() {
            let prev_phase = &self.prev_phase_map[i];
            let prev_synth = &self.prev_synth_map[i];
            let curr_freq = &frame[i];

        }
        processed_frame
    }
}