from typing import Dict, List, Optional, Any, Union
import numpy as np
from numpy.typing import NDArray

# Mixer class implemented in Rust
class Mixer:
    # Initializes Mixer
    def __init__(self) -> None:...

    # Loads preprocessed songs into song dictionary
    def load_preprocessed_song(
        self,
        song_name: str,
        track_name: str,
        data: NDArray[np.float64],
    ) -> None:...

    # Returns list of songs loaded into song dictionary
    def get_song_list(self) -> List[str]:...

    # Prints song dictionary (used for debugging)
    def print_song_map(self) -> None:...

    # Loads track/stem into specified channel
    def load_track(self, title: str, channel: int) -> bool:...

    # Turns channel "on" or "off"
    def channel_on_off(self, channel: int) -> bool:...

    # Adjusts volume of drum, bass, melody, or vocal channels
    def adj_track_vol(self, track: str, adjustment: float) -> None:...

    # Returns volume of specified channel type
    def get_track_vol(self, track: str) -> float:...

    # Returns list of on/off values for stems loaded into channels
    def get_channel_list_on_off(self) -> List[bool]:...

    # Begins playback
    def play(self) -> None:...

    # Stops playback
    def stop(self) -> None:...