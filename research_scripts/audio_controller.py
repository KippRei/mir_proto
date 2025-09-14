import pygame
class AudioPlayer():
    def __init__(self):
        self.init_tracks()

    def init_tracks(self):
        # add stems to buttons
        pygame.mixer.init(channels=4)

        self.drum_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_drums.mp3')
        self.string_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_strings.mp3')
        self.string_track2 = pygame.mixer.Sound('doechii_maybe/doechii_maybe_no_one_mel_pitch.mp3')
        self.vocal_track = pygame.mixer.Sound('doechii_maybe/doechii_maybe_vocals.mp3')

        self.channel1 = pygame.mixer.Channel(0)
        self.channel2 = pygame.mixer.Channel(1)
        self.channel3 = pygame.mixer.Channel(2)
        self.channel4 = pygame.mixer.Channel(3)
        self.channel5 = pygame.mixer.Channel(4)


        self.channel1.play(self.drum_track)
        self.channel2.play(self.string_track)
        self.channel3.play(self.vocal_track)
        # channel4.play(self.drum_track)
        self.channel5.play(self.string_track2)
        self.channel5.set_volume(0)

    def update_vol(self, track, value):
        val = int(value) / 126.0
        match track:
            case 'drums':
                if self.ch1_on:
                    self.channel1.set_volume(float(val))
                    # self.s1.set(val)

            case 'melody':
                if self.ch2_on:
                    self.channel2.set_volume(float(val))
                    # self.s2.set(val)
                else:
                    self.channel5.set_volume(float(val))
                    # self.s2.set(val)
                

            case 'vocals':
                if self.ch3_on:
                    self.channel3.set_volume(float(val))
                    # self.s3.set(val)
                
            
            case _:
                print('Invalid volume adjustment')

class AudioController():
    def __init__(self):
        self.channel1 = self.channel2 = self.channel3 = self.channel4 = self.channel5 = None
        self.drum_track = self.string_track = self.string_track2 = self.vocal_track = None
        self.ch1_on = self.ch2_on = self.ch3_on = self.ch4_on = self.ch5 = None
        # playing bool
        self.ch1_on = True
        self.ch2_on = True
        self.ch3_on = True
        self.ch4_on = True
        self.ch5_on = False

    

    def play_drums(self):
        if not self.ch1_on:
            self.channel1.set_volume(self.s1.get())
            self.button1.config(text='Drums (on)')
            self.ch1_on = True
        else:
            self.channel1.set_volume(0)
            self.button1.config(text='Drums (off)')
            self.ch1_on = False

    def play_melody(self, tr):
        match tr:
            # Channel 2
            case 1:
                if not self.ch2_on:
                    self.channel2.set_volume(self.s2.get())
                    self.channel5.set_volume(0)
                    self.ch2_on = True
                    self.ch5_on = False
                else:
                    self.channel2.set_volume(0)
                    self.ch2_on = False
            
            # Channel 5
            case 2:
                if not self.ch5_on:
                    self.channel5.set_volume(self.s2.get())
                    self.channel2.set_volume(0)
                    self.ch2_on = False
                    self.ch5_on = True
                else:
                    self.channel5.set_volume(0)
                    self.ch5_on = False

    def play_vocals(self):
        if not self.ch3_on:
            self.channel3.set_volume(self.s3.get())
            self.button3.config(text='Vocals (on)')
            self.ch3_on = True
        else:
            self.channel3.set_volume(0)
            self.button3.config(text='Vocals (off)')
            self.ch3_on = False

    # def start_drum_hit(e):
    #     print(e)
    #     self.channel1.set_volume(0)
    #     self.channel4.play(self.drum_track)

    # def stop_drum_hit(e):
    #     self.channel1.set_volume(1)
    #     self.channel4.stop()
