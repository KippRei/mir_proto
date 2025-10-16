use ndarray::Array;
use pyo3::prelude::*;
use cpal::{Data, FromSample, OutputCallbackInfo, Sample, SampleFormat};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use pyo3::types::PyList;
use core::f32::consts::PI;
use core::time;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{f32, thread, vec};
use numpy::ndarray::{Array2, Ix2};
use numpy::{PyReadonlyArray};

static mut PHASE: f32 = 0f32;

struct PlaybackState {
    data: Array2<f32>,
    curr_frame: usize
}

#[pyclass]
pub struct Mixer {
    is_playing: Arc<Mutex<bool>>,
    song_map: HashMap<String, HashMap<String, Arc<Array2<f32>>>>,
    channel_map: Vec<_Channel>,
    playback: Arc<Mutex<PlaybackState>>,
    song_list: Vec<String>,
    track_volumes: HashMap<String, f32>
}

#[pymethods]
impl Mixer {
    #[new]
    pub fn new() -> Self {
        Mixer {
            is_playing: Arc::new(Mutex::new(false)),
            song_map: HashMap::new(),
            channel_map: Mixer::_init_channels(16),
            playback: Arc::new(Mutex::new(PlaybackState { data: Array2::zeros((4351282 * 2, 2)), curr_frame: 0 })),
            song_list: Vec::<String>::new(),
            track_volumes: Mixer::_init_track_volumes()
        }
    }

    pub fn load_preprocessed_song(
        &mut self,
        song_name: String,
        track_name: String,
        data: PyReadonlyArray<f32, Ix2>
    ) -> PyResult<()> {
            if !self.song_list.contains(&song_name) {
                self.song_list.push(song_name.clone());
            }
            let rust_arr = Arc::new(data.as_array().to_owned());
            let outer_song_map = self.song_map
                .entry(song_name)
                .or_insert_with(HashMap::new);
            outer_song_map.insert(track_name, rust_arr);
            self.print_song_map();
            Ok(())
    }  

    // Note: For testing playback
    pub fn add_to_playback(&self, name: String) {
        let song_info = self.song_map.get(&name).unwrap();
        let track = song_info.get("drum").clone().unwrap();
        let playback_clone = self.playback.clone();
        let mut playback_lock = playback_clone.lock().unwrap(); 
        let playback_data = &mut playback_lock.data;
        let mut count_idx: usize = 0;
        for i in track.iter() {
            if count_idx % 2 == 0 {
                playback_data[[count_idx, 0]] = *i;
            }
            count_idx += 1;
        }
    } 

    pub fn get_song_list(&self) -> PyResult<Vec<String>> {
        Ok(self.song_list.clone())
    }

    // For debugging
    pub fn print_song_map(&self) {
        // Iterate over the (key, value) pairs in the map
        for (song_name, track_map) in self.song_map.iter() {
            // Print the Song Name header
            let mut info = String::from("--- Mixer Channel Map Info ---\n");
            info.push_str(&format!("SONG: '{}'\n", song_name));
            
            // INNER LOOP: Iterate over (track_name, array)
            for (track_name, array) in track_map.iter() {
                let shape = array.shape();
                let shape_str = format!("({}, {})", shape[0], shape[1]);
                
                // Print the track information indented
                info.push_str(&format!(
                    "  - Track: '{}' | Shape: {} | Data Type: f64\n",
                    track_name, shape_str
                ));
            }
        println!("{}", info);
        }
    }

    pub fn load_track(&mut self, title: String, channel: i32) -> PyResult<bool> {
        // Types of tracks
        let TRACK_TYPES = ["drum", "bass", "melody", "vocal"];
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
        // Get the channel info from the channel_map
        let Some(channel_to_load) = self.channel_map.get_mut(channel_index) else {
            return Ok(false);
        };
        // Get the track data from the song data
        let Some(track_data) = song.get(track_type) else {
            return Ok(false);
        };
        // Load the track
        channel_to_load.load_data(track_data.clone());
        Ok(true)
    }

    pub fn channel_on_off(&mut self, channel: i32) -> PyResult<bool>{
        let Ok(idx) = usize::try_from(channel) else {
            return Ok(false);
        };
        self.channel_map[idx].is_playing = !self.channel_map[idx].is_playing;
        let col_mod = idx % 4;
        for i in 0..15 {
            if i % 4 == col_mod && i != idx {
                self.channel_map[i].is_playing = false;
            }
        }
        Ok(true)
    }

    pub fn adj_track_vol(&mut self, track: String, adjustment: f32) {
        if let Some(track_vol) = self.track_volumes.get_mut(&track) {
            *track_vol += adjustment;
            if *track_vol > 1.0 {*track_vol = 1.0}
            else if *track_vol < 0.0 {*track_vol = 0.0}                
        }
    }

    pub fn get_track_vol(&self, track: String) -> PyResult<f32>{
        let Some(track_vol) = self.track_volumes.get(&track) else {
            return Ok(0.0);
        };
        Ok(*track_vol)
    }

    pub fn get_channel_list_on_off(&self) -> PyResult<Vec<bool>> {
        let mut return_list = Vec::new();
        for val in &self.channel_map {
            return_list.push(val.is_playing);
        }
        Ok(return_list)
    }

    pub fn play(&self) {
        let mut is_playing_lock = self.is_playing.lock().unwrap();
        *is_playing_lock = true;
        Mixer::_play(self);
    }

    pub fn stop(&self) {
        let mut is_playing_lock = self.is_playing.lock().unwrap();
        *is_playing_lock = false;
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

    fn _init_track_volumes() -> HashMap<String, f32> {
        let TRACK_TYPES = ["drum", "bass", "melody", "vocal"];
        let mut new_vol_map = HashMap::new();
        for track_type in TRACK_TYPES {
            new_vol_map.insert(track_type.to_string(), 1f32);
        }
        new_vol_map
    }

    pub fn _play(&self) {
        let host = cpal::default_host();
        let device = host.default_output_device().expect("No output device available.");
        let mut supported_configs_range = device.supported_output_configs().expect("Error querying configs.");
        let supported_config = supported_configs_range.find(|config| {
            matches!(
                config.sample_format(),
                cpal::SampleFormat::F32 | cpal::SampleFormat::I16 | cpal::SampleFormat::U16
            )
        }).expect("No valid sample format found.");
        let complete_config = supported_config.with_max_sample_rate();
        let sample_format = supported_config.sample_format();
        let config: cpal::StreamConfig = complete_config.into();
        let err_fn = |err| eprintln!("Error occured while trying to play: {}", err);

        let is_playing_clone = self.is_playing.clone();
        let playback_clone = self.playback.clone();

        thread::spawn(move || {
            let stream = match sample_format {
                SampleFormat::F32 => device.build_output_stream(&config, move |data: &mut [f32],  _: &cpal::OutputCallbackInfo| {
                    let mut playback_state = playback_clone.lock().unwrap();
                    Mixer::real_time_audio(data, &mut playback_state);
                }, err_fn, None),
                SampleFormat::I16 => device.build_output_stream(&config, move |data: &mut [i16],  _: &cpal::OutputCallbackInfo| {
                    let mut playback_state = playback_clone.lock().unwrap();
                    Mixer::real_time_audio(data, &mut playback_state);
                }, err_fn, None),
                SampleFormat::U16 => device.build_output_stream(&config, move |data: &mut [u16],  _: &cpal::OutputCallbackInfo| {
                    let mut playback_state = playback_clone.lock().unwrap();
                    Mixer::real_time_audio(data, &mut playback_state);
                }, err_fn, None),
                sample_format => panic!("Unsupported sample format '{sample_format}'")
            }.unwrap();

            stream.play().expect("Error playing.");

            while *is_playing_clone.lock().unwrap() {
                thread::sleep(Duration::from_millis(10));
            }
        });
    }

    fn real_time_audio<T: Sample> (data: &mut[T], playback_state: &mut PlaybackState)
    where T: Sample + FromSample<f32> {
        let playback_state_data = &playback_state.data;
        let mut curr_frame = playback_state.curr_frame;
        for out_frame in data.chunks_mut(2) {
            let left = playback_state_data[[curr_frame, 0]];
            let right = playback_state_data[[curr_frame, 1]];

            out_frame[0] = T::from_sample(left);
            out_frame[1] = T::from_sample(right);

            curr_frame += 1;
        }
        playback_state.curr_frame = curr_frame;
    }

}

struct _Channel {
    is_playing: bool,
    volume: f32,
    data: Option<Arc<Array2<f32>>>,
}

impl _Channel {
    fn new() -> _Channel {
        _Channel {
            is_playing: false,
            volume: 1.0,
            data: None
        }
    }

    fn load_data(&mut self, new_data: Arc<Array2<f32>>) {
        self.data = Some(new_data);
    }

    fn is_playing(&self) -> bool {
        self.is_playing
    }

    fn get_volume(&self) -> f32 {
        self.volume
    }
}