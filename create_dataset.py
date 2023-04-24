import glob

from recording import Recording

path = '/home/faren/Documents/HB/Dataset/mit-bih-arrhythmia-database-1.0.0'
dest = '/home/faren/Documents/HB/Beats/'
seg_len = 300
lead = 'MLII'

for file in glob.iglob(path + '/*.dat'):
    record = Recording(file, lead)
    if hasattr(record, 'signal'):
        record.segment_beats(seg_len=seg_len)
        record.save_beats(dest + record.name)
        record.load_beats(dest + record.name)