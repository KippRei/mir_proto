use pyo3::prelude::*;

mod audio_player;

use audio_player::Mixer;

#[pymodule]
fn zmkr_audio_engine(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mixer>()?;

    Ok(())
}
