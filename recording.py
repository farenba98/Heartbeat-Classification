import numpy as np
import os
import glob
import h5py
import wfdb

class Recording:
    def __init__(self, name = ''):
        self.name = name
        self.signal = []
        self.ann_samples = []
        self.ann_symbols = []
        self.beats = []

    def read_data(self, file, lead):
        self.name = file.rpartition('/')[-1].split('.')[0]
        path = file.rpartition('/')[0]
        file_path = f"{path}/{self.name}"
        record = wfdb.rdrecord(file_path)
        if not lead in record.sig_name:
            return None
        self.signal = record.p_signal[:, record.sig_name.index(lead)]
        ann = wfdb.rdann(file_path, 'atr')
        self.ann_samples = ann.sample
        self.ann_symbols = ann.symbol


    def segment_beats(self, beat_types = [], seg_len = 300):
        for i, peak in enumerate(self.ann_samples):
            beat_type = self.ann_symbols[i]
            if peak > seg_len/2 and peak + seg_len/2 < len(self.signal) and (beat_type in beat_types or beat_types == []):
                segment = self.signal[int(peak-seg_len/2):int(peak+seg_len/2)]
                beat = {'type': beat_type, 'segment': segment}
                self.beats.append(beat)
    
    def save_beats(self, dest_path):
        os.makedirs(dest_path, exist_ok = True)
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
                beat = {'type': beat_type, 'signal': segment}
                self.beats.append(beat)