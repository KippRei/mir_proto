<div>
A program that allows a user to easily mix/mashup songs using a MIDI controller (the PreSonus Atom). The user selected songs will be preprocessed and, once preprocessing is completed, the user will be able to select stems of the different 
tracks (vocals, melody, bass, drums) and mix them together in real time. The overall mix will loop to allow for continuous mixing/playing of the music. The software handles: splitting the mp3 file into stems, BPM detection and beat 
matching, and simple structure analysis to attempt to find a good start point for the stems.<br />
<br />
NOTES:<br />
-This is not a release version, it is merely intended to show my progress.<br />
-In the prototype phase, the software chooses a song/phrase start point by determining when drums enter the mix. The next phase (Dec./Jan.), I will begin working on a ML model for structure analysis to make better decisions for song/phrase
 starting point.<br />
</div>
<br />
<br />
Step 1: Select songs for preprocessing by clicking button to open file explorer or dragging and dropping onto button.
<br />
<br />
<div align="center">
<img width="50%" alt="Step 1 Image" src="https://github.com/user-attachments/assets/e56a9f50-1d2c-47af-a0e3-fdb8bec4935c" />
</div>
<br />
<br />
Step 2: Once songs are done preprocessing, the song list will refresh with available songs.
<br />
<br />
<div align="center">
<img width="50%" alt="Step 2 Image" src="https://github.com/user-attachments/assets/a613217a-fb3e-4768-8c0d-8ecd3358693a" />
</div>
<br />
<br />
Step 3: Drag songs from song list to pad to load desired stem (i.e. drums, bass, melody, vocals). Songs can be loaded onto any of the 4 pads available for desired stem (i.e. up to 4 stem types can be loaded at a time).
<br />
<br />
<div align="center">
<img width="50%" alt="Step 3 Image 1" src="https://github.com/user-attachments/assets/0d5324be-ff1d-44f4-af11-0c88b4cd03c6" />
 <br />
 <br />
<img width="50%" alt="Step 3 Image 2" src="https://github.com/user-attachments/assets/c7be839a-d091-46ea-a243-2111f31b0598" />
</div>
<br />
<br />
Step 4: Press the corresponding pad on the USB MIDI controller to turn on stem [1], then press play on controller to start the loop (currently 32 bars) [2]. Stems can be loaded, swapped, and turned on and off in real-time. Individual stem volume can be controlled using the four corresponding knobs on the controller [3].
<br />
<br />
<div align="center">
<img width="50%" alt="Step 4 Image 1" src="https://github.com/user-attachments/assets/79c9b26f-5463-415d-b93b-d086878bc3d1" />
  <br />
 <br />
<img width="50%" alt="Step 4 Image 2" src="https://github.com/user-attachments/assets/befd34aa-e301-43cb-8c6b-d6abedc14878" />
</div>
