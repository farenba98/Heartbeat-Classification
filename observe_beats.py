from recording import Recording

dest = '/home/faren/Documents/HB/Beats/'

record_name = '233'
record = Recording(record_name)
record.load_beats(dest + record.name)