import numpy as np
import sounddevice as sd
import queue

def record_fixed_duration(duration, sample_rate=44100, channels=1, device_index=None):
    """
    Records audio for a set duration and blocks execution until complete.
    
    Parameters:
        duration (float): Length of the recording in seconds.
        sample_rate (int): Target sample rate.
        channels (int): Number of channels (1 for mono, 2 for stereo).
        device_index (int): The hardware device index (from hardware.py).
        
    Returns:
        tuple: (audio_array, sample_rate)
    """
    print(f"Recording {duration} seconds...")
    
    # sd.rec starts recording in the background
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        device=device_index,
        dtype='float64'
    )
    
    # sd.wait blocks the script until the recording is finished
    sd.wait()
    
    # If mono, flatten the array from shape (N, 1) to (N,)
    if channels == 1:
        recording = recording.flatten()
        
    return recording, sample_rate


class ContinuousRecorder:
    """
    A non-blocking audio recorder designed to be integrated into a GUI.
    It captures audio into an internal queue until explicitly stopped.
    """
    def __init__(self, sample_rate=44100, channels=1, device_index=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False

    def _audio_callback(self, indata, frames, time, status):
        """
        This callback is called by PortAudio for every block of audio captured.
        """
        if status:
            print(f"Hardware Status: {status}")
        # We must copy the indata because the buffer is overwritten by the system
        self.audio_queue.put(indata.copy())

    def start(self):
        """
        Opens the hardware stream and begins capturing audio in a background thread.
        """
        # Clear any old data from the queue
        self.audio_queue = queue.Queue()
        self.is_recording = True
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device_index,
            channels=self.channels,
            callback=self._audio_callback,
            dtype='float64'
        )
        self.stream.start()
        print("Live recording started...")

    def stop(self):
        """
        Stops the stream, empties the queue, and concatenates the blocks into a final array.
        
        Returns:
            tuple: (audio_array, sample_rate)
        """
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            print("Live recording stopped.")

        frames = []
        while not self.audio_queue.empty():
            frames.append(self.audio_queue.get())

        if not frames:
            print("Warning: No audio frames captured.")
            return np.array([]), self.sample_rate

        # Stack all the tiny audio blocks into one continuous array
        audio_array = np.concatenate(frames, axis=0)
        
        if self.channels == 1:
            audio_array = audio_array.flatten()

        return audio_array, self.sample_rate