import glob

from recording import Recording

path = '/home/faren/Documents/HB/Dataset/mit-bih-arrhythmia-database-1.0.0'
seg_len = 300
lead = 'MLII'

for file in glob.iglob(path + '/*.dat'):
    record = Recording(file, lead)
    if hasattr(record, 'signal'):
        print(record.record_name)
        record.segment_beats(seg_len=seg_len)
        record.save_beats('/home/faren/Documents/HB/Beats/' + record.record_name)
        record.load_beats('/home/faren/Documents/HB/Beats/' + record.record_name)