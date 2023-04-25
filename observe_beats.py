import matplotlib.pyplot as plt
import pandas as pd

from recording import Recording

dest = '/home/faren/Documents/HB/Beats/'

record_name = '233'
record = Recording(record_name)
record.load_beats(dest + record.name)

print(record.beats[0]['type'])

plt.plot(record.beats[500]['signal'])
plt.show()