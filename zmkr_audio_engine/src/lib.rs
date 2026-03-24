use pyo3::prelude::*;

mod audio_player;
mod phase_vocoder;
mod jneem_pv;

use audio_player::Mixer;

#[pymodule]
fn zmkr_audio_engine(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mixer>()?;

    Ok(())
}
