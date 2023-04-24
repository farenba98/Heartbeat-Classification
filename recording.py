import os
import glob
import h5py
import wfdb

class Recording:
    def __init__(self, file, lead):
        self.file = file
        self.record_name = file.rpartition('/')[-1].split('.')[0]
        self.path = file.rpartition('/')[0]
        self.file_path = f"{self.path}/{self.record_name}"

        self.record = wfdb.rdrecord(self.file_path)
        if not lead in self.record.sig_name:
            return None
        self.signal = self.record.p_signal[:, self.record.sig_name.index(lead)]
        self.ann = wfdb.rdann(self.file_path, 'atr')
        self.ann_samples = self.ann.sample
        self.ann_symbols = self.ann.symbol
        self.beats = []
    
    def segment_beats(self, seg_len=300):
        for i, peak in enumerate(self.ann_samples):
            beat_type = self.ann_symbols[i]
            if peak > 150 and beat_type in ["N", "S", "V", "F", "Q"]:
                segment = self.signal[int(peak-seg_len/2):int(peak+seg_len/2)]
                beat = {'type': beat_type, 'segment': segment}
                self.beats.append(beat)
    
    def save_beats(self, dest_path):
        os.makedirs(dest_path, exist_ok=True)
        for i, beat in enumerate(self.beats):
            beat_type = beat['type']
            segment = beat['segment']
            filename = os.path.join(dest_path, f'{i}_{beat_type}.h5')
            with h5py.File(filename, 'w') as hf:
                hf.create_dataset('signal', data=segment)
    
    def load_beats(self, dest_path):
        self.beats = []
        for file in glob.iglob(dest_path + '/*.h5'):
            with h5py.File(file, 'r') as hf:
                segment = np.array(hf['signal'])
                beat_type = os.path.splitext(os.path.basename(file))[0].split('_')[-1]
                beat = {'type': beat_type, 'segment': segment}
                self.beats.append(beat)
