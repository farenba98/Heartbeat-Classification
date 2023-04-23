import numpy as np
import matplotlib.pyplot as plt
import wfdb

path = '/home/faren/Documents/HB/Dataset/mit-bih-arrhythmia-database-1.0.0'
record_name = '100'
file_path = f"{path}/{record_name}"

record = wfdb.rdrecord(file_path)
signal = record.p_signal[:, record.sig_name.index('MLII')]
ann = wfdb.rdann(file_path, 'atr')
ann_samples = ann.sample
ann_symbols = ann.symbol

# wfdb.plot_items(signal=signal, ann_samp=[ann_samples], ann_sym=[ann_symbols])
# plt.savefig('sig.png')

segmented_beats = []
seg_len = 300

for i, peak in enumerate(ann_samples):
    beat_type = ann_symbols[i]
    if peak > 150 and beat_type in ["N", "S", "V", "F", "Q"]:
        segment = signal[int(peak-seg_len/2):int(peak+seg_len/2)]
        # Append segment and corresponding type to segmented_beats
        segmented_beats.append((segment, beat_type))

first_segment, first_label = segmented_beats[0]
print("Signal segment:", first_segment)
print("Label:", first_label)