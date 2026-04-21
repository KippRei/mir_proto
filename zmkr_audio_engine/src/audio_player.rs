use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat};
use numpy::ndarray::{Array2, Array3, ArrayBase, Ix2};
use numpy::PyReadonlyArray;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::f32::MIN;
use std::io::SeekFrom;
use std::ops::{Add, AddAssign, Deref};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{f32, thread};
use rustfft::{num_complex::Complex};
use ringbuf::{HeapProd, HeapCons, HeapRb, traits::*};

use crate::phase_vocoder::PhaseVocoder;

struct PlaybackState {
    curr_frame: usize,
}

#[pyclass]
/// Mixer implemented in Rust for speed and memory safety
/// Takes music as ndarrays from Python rust_audio_manager.py
pub struct Mixer {
    is_playing: Arc<Mutex<bool>>,
    song_map: HashMap<String, HashMap<String, Arc<Array2<f64>>>>,
    channel_map: Arc<Mutex<Vec<_Channel>>>,
    playback: Arc<Mutex<PlaybackState>>,
    song_list: Vec<String>,
    track_volumes: Arc<Mutex<HashMap<String, f64>>>,
    prod_raw: Arc<Mutex<Option<HeapProd<f32>>>>,
    prod_proc: Option<HeapProd<f32>>,
    cons_proc: Option<HeapCons<f32>>,
    pv: Arc<Mutex<PhaseVocoder>>,
}

#[pymethods]
impl Mixer {
    #[new]
    pub fn new() -> Self {
        let (prod_r, cons_r) = HeapRb::<f32>::new(PhaseVocoder::WINDOW_SZ * 4).split();
        let (prod_p, cons_p) = HeapRb::<f32>::new(PhaseVocoder::WINDOW_SZ * 2).split();

        let new_mixer = Mixer {
            is_playing: Arc::new(Mutex::new(false)),
            song_map: HashMap::new(),
            channel_map: Arc::new(Mutex::new(Mixer::_init_channels(16))),
            playback: Arc::new(Mutex::new(PlaybackState { curr_frame: 0 })),
            song_list: Vec::<String>::new(),
            track_volumes: Arc::new(Mutex::new(Mixer::_init_track_volumes())),
            prod_raw: Arc::new(Mutex::new(Some(prod_r))),
            prod_proc: Some(prod_p),
            cons_proc: Some(cons_p),
            pv: Arc::new(Mutex::new(PhaseVocoder::new())),
        };
        new_mixer
    }

    /// Loads songs from Python rust_audio_manager script
    pub fn load_preprocessed_song(
        &mut self,
        song_name: String,
        track_name: String,
        data: PyReadonlyArray<f64, Ix2>,
    ) -> PyResult<()> {
        if !self.song_list.contains(&song_name) {
            self.song_list.push(song_name.clone());
        }
        let rust_arr = Arc::new(data.as_array().to_owned());
        let outer_song_map = self.song_map.entry(song_name).or_insert_with(HashMap::new);
        outer_song_map.insert(track_name, rust_arr);
        Ok(())
    }

    /// Starts PV and begins processing audio data from ring buffer
    pub fn start_audio_processing(&mut self) {
        let prod_raw_arc = self.prod_raw.clone();
        let playback_clone = self.playback.clone();
        let channel_map_clone = self.channel_map.clone();
        let volume_clone = self.track_volumes.clone();

        thread::spawn(move || {
            let mut data: [f32; 512] = [0f32; 512];

            loop {
                let mut prod_lock = prod_raw_arc.lock().unwrap();
                if let Some(prod) = prod_lock.as_mut() {
                    if prod.vacant_len() >= data.len() {
                        let channel_map_lock = channel_map_clone.lock().unwrap();
                        let mut playback_state = playback_clone.lock().unwrap();
                        let volume_lock = volume_clone.lock().unwrap();
                        let volume_ref = &volume_lock.deref();

                        Mixer::real_time_audio(&mut data, &mut playback_state, &channel_map_lock, volume_ref);
                        prod.push_slice(&data);
                    }
                }
                thread::sleep(Duration::from_millis(1));
            }
        });
    }

    // Note: For testing playback
    // pub fn add_to_playback(&self, name: String, track_type: String) {
    //     let song_info = self.song_map.get(&name).unwrap();
    //     let track = song_info.get(&track_type).clone().unwrap();

    //     // let playback_clone = self.playback.clone();
    //     let mut playback_lock = self.playback.lock().unwrap();
    //     let playback_data = &mut playback_lock.data;
    //     let mut count_idx: usize = 0;
    //     let mut sample_idx: usize;
    //     let mut channel_idx: usize;
    //     for i in track.iter() {
    //         sample_idx = count_idx / 2;
    //         channel_idx = count_idx % 2;
    //         playback_data[[sample_idx, channel_idx]] += *i;

    //         count_idx += 1;
    //     }
    // }

    /// Gets the song list (list of playing song)
    pub fn get_song_list(&self) -> PyResult<Vec<String>> {
        Ok(self.song_list.clone())
    }

    // // For debugging
    // pub fn print_song_map(&self) {
    //     // Iterate over the (key, value) pairs in the map
    //     for (song_name, track_map) in self.song_map.iter() {
    //         // Print the Song Name header
    //         let mut info = String::from("--- Mixer Channel Map Info ---\n");
    //         info.push_str(&format!("SONG: '{}'\n", song_name));

    //         // INNER LOOP: Iterate over (track_name, array)
    //         for (track_name, array) in track_map.iter() {
    //             let shape = array.shape();
    //             let shape_str = format!("({}, {})", shape[0], shape[1]);

    //             // Print the track information indented
    //             info.push_str(&format!(
    //                 "  - Track: '{}' | Shape: {} | Data Type: f64\n",
    //                 track_name, shape_str
    //             ));
    //         }
    //         println!("{}", info);
    //     }
    // }

    /// Loads stem when button pressed on MIDI controller
    pub fn load_track(&mut self, title: String, channel: i32) -> PyResult<bool> {
        // Types of tracks
        const TRACK_TYPES: [&'static str; 4] = ["drum", "bass", "melody", "vocal"];
        // Get the channel index (channel number converted to usize)
        let Ok(channel_index) = usize::try_from(channel) else {
            return Ok(false);
        };
        // Get the track type
        let track_type = TRACK_TYPES[channel_index % 4];
        // Get the song info from the song_map
        let Some(song) = self.song_map.get(&title) else {
            return Ok(false);
        };
        let mut channel_map_lock = self.channel_map.lock().unwrap();
        // Get the channel info from the channel_map
        let Some(channel_to_load) = channel_map_lock.get_mut(channel_index) else {
            return Ok(false);
        };
        // Get the track data from the song data
        let Some(track_data) = song.get(track_type) else {
            return Ok(false);
        };
        // Set track name
        channel_to_load.set_name(track_type.to_string());
        // Load the track
        channel_to_load.load_data(track_data.clone());
        Ok(true)
    }

    pub fn channel_on_off(&mut self, channel: i32) -> PyResult<bool> {
        let Ok(idx) = usize::try_from(channel) else {
            return Ok(false);
        };
        let mut channel_map_lock = self.channel_map.lock().unwrap();
        channel_map_lock[idx].is_playing = !channel_map_lock[idx].is_playing;
        let col_mod = idx % 4;
        for i in 0..16 {
            if i % 4 == col_mod && i != idx {
                channel_map_lock[i].is_playing = false;
            }
        }
        Ok(true)
    }

    /// Adjusts stem type volume
    pub fn adj_track_vol(&mut self, track: String, adjustment: f64) {
        let mut track_vol_lock = self.track_volumes.lock().unwrap();
        if let Some(track_vol) = track_vol_lock.get_mut(&track) {
            *track_vol += adjustment;
            if *track_vol > 1.0 {
                *track_vol = 1.0
            } else if *track_vol < 0.0 {
                *track_vol = 0.0
            }
        }
    }

    pub fn get_track_vol(&self, track: String) -> PyResult<f64> {
        let track_vol_lock = self.track_volumes.lock().unwrap();
        let Some(track_vol) = track_vol_lock.get(&track) else {
            return Ok(0.0);
        };
        Ok(*track_vol)
    }

    pub fn get_channel_list_on_off(&self) -> PyResult<Vec<bool>> {
        let mut return_list = Vec::new();
        let channel_map_lock = self.channel_map.lock().unwrap();
        let channel_map_ref = channel_map_lock.deref();
        for val in channel_map_ref {
            return_list.push(val.is_playing);
        }
        Ok(return_list)
    }

    pub fn play(&mut self) {
        let (new_prod_raw, mut new_cons_raw) = HeapRb::<f32>::new(PhaseVocoder::WINDOW_SZ * 8).split();
        let (mut new_prod_processed, new_cons_processed) = HeapRb::<f32>::new(PhaseVocoder::WINDOW_SZ * 8).split();
        
        {
            let mut prod_lock = self.prod_raw.lock().unwrap();
            *prod_lock = Some(new_prod_raw);
        }

        let is_playing_clone = self.is_playing.clone();
        // Set is_playing flag to indicate active playback
        *is_playing_clone.lock().unwrap() = true;

        let pv_clone = self.pv.clone();

        // PV thread
        thread::spawn(move || {
            let mut acc_buf = vec![0.0f32; PhaseVocoder::WINDOW_SZ * 2];
            let mut sample_count = 0;
            
            while *is_playing_clone.lock().unwrap() {
                while sample_count < acc_buf.len() {
                    if let Some(sample) = new_cons_raw.try_pop() {
                        acc_buf[sample_count] = sample;
                        sample_count += 1;
                    } else { break; }
                }

                if sample_count == acc_buf.len() {
                    let mut pv = pv_clone.lock().unwrap();
                    pv.pv_run(&acc_buf, 124);
                    while new_prod_processed.vacant_len() < pv.output_buf.len() {
                        thread::sleep(Duration::from_millis(1));
                    }
                    new_prod_processed.push_slice(&pv.output_buf);
                    sample_count = 0;
                }
                thread::sleep(Duration::from_millis(10));
            }
        });

        Mixer::_play(self, new_cons_processed);
    }

    pub fn stop(&mut self) {
        let mut is_playing_lock = self.is_playing.lock().unwrap();
        let mut playback_state = self.playback.lock().unwrap();
        playback_state.curr_frame = 0;
        *is_playing_lock = false;

        *self.prod_raw.lock().unwrap() = None;
    }
}

impl Mixer {
    fn _init_channels(num_of_channels: i32) -> Vec<_Channel> {
        let mut channel_map = Vec::<_Channel>::new();
        for _ in 0..num_of_channels {
            let new_channel = _Channel::new();
            channel_map.push(new_channel);
        }
        channel_map
    }

    fn _init_track_volumes() -> HashMap<String, f64> {
        const TRACK_TYPES: [&'static str; 4] = ["drum", "bass", "melody", "vocal"];
        let mut new_vol_map = HashMap::new();
        for track_type in TRACK_TYPES {
            new_vol_map.insert(track_type.to_string(), 1f64);
        }
        new_vol_map
    }

    /// Starts separate thread for playing of music (sends samples to audio card)
    pub fn _play(&self, mut cons: HeapCons<f32>) {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("No output device available.");
        let mut supported_configs_range = device
            .supported_output_configs()
            .expect("Error querying configs.");
        let supported_config = supported_configs_range
            .find(|config| {
                matches!(
                    config.sample_format(),
                    cpal::SampleFormat::F32 | cpal::SampleFormat::I16 | cpal::SampleFormat::U16
                )
            })
            .expect("No valid sample format found.");
        let complete_config = supported_config.with_max_sample_rate();
        let sample_format = supported_config.sample_format();
        let config: cpal::StreamConfig = complete_config.into();
        let err_fn = |err| eprintln!("Error occured while trying to play: {}", err);

        let is_playing_clone = self.is_playing.clone();
        let playback_clone = self.playback.clone();
        let channel_map_clone = self.channel_map.clone();
        let volume_clone = self.track_volumes.clone();

        thread::spawn(move || {
            let stream = match sample_format {
                SampleFormat::F32 => device.build_output_stream(
                    &config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        Mixer::get_audio(data, &mut cons);
                    },
                    err_fn,
                    None,
                ),
                SampleFormat::I32 => device.build_output_stream(
                    &config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        Mixer::get_audio(data, &mut cons);
                    },
                    err_fn,
                    None,
                ),
                SampleFormat::I16 => device.build_output_stream(
                    &config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        Mixer::get_audio(data, &mut cons);
                    },
                    err_fn,
                    None,
                ),
                sample_format => panic!("Unsupported sample format '{sample_format}'"),
            }
            .unwrap();

            stream.play().expect("Error playing.");

            while *is_playing_clone.lock().unwrap() {
                thread::sleep(Duration::from_millis(10));
            }
        });
    }

    fn real_time_audio (
        data: &mut [f32; 512],
        playback_state: &mut PlaybackState,
        channel_map: &Vec<_Channel>,
        volumes: &HashMap<String, f64>,
    ) 
    // where
    //     T: Sample + FromSample<f32> + AddAssign + std::ops::Add, f64: FromSample<T>
    {
        // TODO: Fix hardcoded 32 bar loop (currently set at 124 bpm)
        let BARS_32: usize = (32f32 * 4f32 * (60f32 / 124f32) * 48000f32).round() as usize;
        // let playback_state_data = &playback_state.data;
        let mut curr_frame = playback_state.curr_frame;

        let active_stems: Vec<(&Array2<f64>, f64)> = channel_map
        .iter()
        .filter(|c| c.is_playing)
        .filter_map(|c| {
            // Dereference the Arc to get the actual Array2 pointer
            c.data.as_ref().map(|data_arc| (data_arc.as_ref(), *volumes.get(&c.name).unwrap_or(&1.0)))
        })
        .collect();

        data.iter_mut().for_each(|s| *s = Sample::EQUILIBRIUM);

        for out_frame in data.chunks_mut(2) {
            let mut left_mix: f64 = 0.0;
            let mut right_mix: f64 = 0.0;

            for (audio_data, vol) in &active_stems {
                // Check bounds once to prevent panics
                if curr_frame < audio_data.shape()[0] {
                    // ndarray indexing is highly optimized pointer arithmetic
                    left_mix += audio_data[[curr_frame, 0]] * vol;
                    right_mix += audio_data[[curr_frame, 1]] * vol;
                }
            }

            left_mix = left_mix.clamp(-1.0, 1.0);
            right_mix = right_mix.clamp(-1.0, 1.0);

            out_frame[0] = left_mix as f32;
            out_frame[1] = right_mix as f32;

            curr_frame += 1;

            if curr_frame == BARS_32 {
                curr_frame = 0;
            }
        }
        
        playback_state.curr_frame = curr_frame;
    }
 
    fn get_audio<T>(data: &mut [T], cons: &mut impl ringbuf::traits::Consumer<Item=f32>) where
        T: Sample + FromSample<f32> {
        data.iter_mut().for_each(|s| *s = Sample::EQUILIBRIUM);
        for out_frame in data.chunks_mut(2) {
            let left = cons.try_pop().unwrap_or(0.0);
            let right = cons.try_pop().unwrap_or(0.0);
            out_frame[0] = T::from_sample_(left);
            out_frame[1] = T::from_sample(right); 
        }
    }
}

// TODO: getters/setters
pub struct _Channel {
    pub name: String,
    pub is_playing: bool,
    pub data: Option<Arc<Array2<f64>>>,
    pub synth_data: Option<Arc<Array2<f64>>>,
    pub curr_frame: usize,
}

// TODO: need to reset curr_frame when new stem is loaded
impl _Channel {
    fn new() -> _Channel {
        _Channel {
            name: "None".to_string(),
            is_playing: false,
            data: None,
            synth_data: None,
            curr_frame: 0,
        }
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing
    }

    pub fn is_loaded(&self) -> bool {
        self.data.is_some()
    }

    fn load_data(&mut self, new_data: Arc<Array2<f64>>) {
        self.data = Some(new_data);
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
}