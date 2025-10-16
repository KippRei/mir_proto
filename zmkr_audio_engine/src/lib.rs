use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::{pyfunction, pymodule, wrap_pyfunction}; 
use cpal::{Data, FromSample, OutputCallbackInfo, Sample, SampleFormat};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use core::f64::consts::PI as PPI;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;
use std::{thread, vec};
use numpy::ndarray::{Array2, Ix2, ArrayView};
use numpy::{PyReadonlyArray};

#[pyclass]
pub struct Mixer {
    is_playing: bool,
    song_map: HashMap<String, HashMap<String, Arc<Array2<f64>>>>,
    channel_map: Vec<_Channel>,
    current_frame: usize,
    song_list: Vec<String>
}

#[pymethods]
impl Mixer {
    #[new]
    pub fn new() -> Self {
        Mixer {
            is_playing: false,
            song_map: HashMap::new(),
            channel_map: Mixer::_init_channels(16),
            current_frame: 0,
            song_list: Vec::<String>::new()

        }
    }

    pub fn load_preprocessed_song(
        &mut self,
        song_name: String,
        track_name: String,
        data: PyReadonlyArray<f64, Ix2>
    ) -> PyResult<()> {
            if !self.song_list.contains(&song_name) {
                self.song_list.push(song_name.clone());
            }
            let rust_arr = Arc::new(data.as_array().to_owned());
            let outer_song_map = self.song_map
                .entry(song_name)
                .or_insert_with(HashMap::new);
            outer_song_map.insert(track_name, rust_arr);

            Ok(())
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
        if !self.song_map.contains_key(&title) {
            return Ok(false)
        } 

        let song = &self.song_map[&title];
        match channel {
            0 => {
                if song.contains_key("drum") {
                    self.channel_map[0].load_data(song["drum"].clone());
                    return Ok(true)
                }
                else {
                    return Ok(false)
                }
            }

            _ => {
                Ok(false)
            }
        }
    }

    pub fn play(&mut self) {
        // Play selected tracks
    }
}

impl Mixer {
    fn _init_channels(num_of_channels: i32) -> Vec<_Channel> {
        let mut channel_map = Vec::<_Channel>::new();
        for i in 0..num_of_channels {
            let new_channel = _Channel::new();
            channel_map.push(new_channel);
        }
        channel_map
    }
}

struct _Channel {
    is_playing: bool,
    volume: f32,
    data: Option<Arc<Array2<f64>>>,
}

impl _Channel {
    fn new() -> _Channel {
        _Channel {
            is_playing: false,
            volume: 1.0,
            data: None
        }
    }

    fn load_data(&mut self, new_data: Arc<Array2<f64>>) {
        self.data = Some(new_data);
    }

    fn is_playing(&self) -> bool {
        self.is_playing
    }

    fn get_volume(&self) -> f32 {
        self.volume
    }
}

#[pymodule]
fn zmkr_audio_engine(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mixer>()?; 

    Ok(())
}