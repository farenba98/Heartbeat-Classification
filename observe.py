import wfdb
import matplotlib.pyplot as plt

from recording import Recording

path = '/home/faren/Documents/HB/Dataset/mit-bih-arrhythmia-database-1.0.0'
dest = '/home/faren/Documents/HB/Beats/'
seg_len = 300
lead = 'MLII'

record_name = '233'
record = Recording(record_name)
file = f"{path}/{record_name}.dat"
record.read_data(file, lead)
if hasattr(record, 'signal'):
    wfdb.plot_items(signal = record.signal, ann_samp = [record.ann_samples], ann_sym = [record.ann_symbols])
    plt.show()