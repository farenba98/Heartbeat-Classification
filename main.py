import matplotlib.pyplot as plt
import wfdb
import glob
import os

seg_len = 300
lead = 'MLII'
path = '/home/faren/Documents/HB/Dataset/mit-bih-arrhythmia-database-1.0.0'

for file in glob.iglob(path + '/*.dat'):
    record_name = file.rpartition('/')[-1].split('.')[0]
    file_path = f"{path}/{record_name}"
    record = wfdb.rdrecord(file_path)
    if not lead in record.sig_name:
        continue 
    signal = record.p_signal[:, record.sig_name.index(lead)]
    ann = wfdb.rdann(file_path, 'atr')
    ann_samples = ann.sample
    ann_symbols = ann.symbol

    wfdb.plot_items(signal=signal, ann_samp=[ann_samples], ann_sym=[ann_symbols])
    plt.savefig('sig.png')

    os.mkdir('/home/faren/Documents/HB/Beats/' + record_name)
    for i, peak in enumerate(ann_samples):
        beat_type = ann_symbols[i]
        if peak > seg_len/2 and beat_type in ["N", "S", "V", "F", "Q"]:
            segment = signal[int(peak-seg_len/2):int(peak+seg_len/2)]
            dest_path = '/home/faren/Documents/HB/Beats/' + record_name + '/' + str(i) + '_' + str(beat_type) + '.h5'
            with h5py.File(dest_path, 'w') as hf:
                hf.create_dataset('signal',  data=segment)
                # hf.create_dataset('label', data=beat_type)

    dest_path = '/home/faren/Documents/HB/Beats/' + record_name + '/'
    for file in glob.iglob(dest_path + '*.h5'):
        with h5py.File(file, 'r') as hf:
            data = np.array(hf['signal'][:])
            label = file.split("_")[1].split(".")[0]
            if label != 'N':
                print(label)