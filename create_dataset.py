import glob
import os

from recording import Recording

path = '/home/faren/Documents/HB/Dataset/mit-bih-arrhythmia-database-1.0.0'
dest = '/home/faren/Documents/HB/Beats/'
seg_len = 300
lead = 'MLII'
valid_labels= ["N", "S", "V", "F", "Q"]

for file in glob.iglob(path + '/*.dat'):
    record = Recording()
    record.read_data(file, lead)
    if hasattr(record, 'signal'):
        record.segment_beats(beat_types = valid_labels, seg_len = seg_len)
        if not os.path.exists(dest + record.name):
            os.mkdir(dest + record.name)
        record.save_beats(dest + record.name)