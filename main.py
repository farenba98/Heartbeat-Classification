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
# plt.show()