import numpy as np
import matplotlib.pyplot as plt
import wfdb

# Set the path to the directory that contains the record files
path = '/home/faren/Documents/Heartbeat Classification/Dataset/mit-bih-arrhythmia-database-1.0.0';

# Set the name of the record file (without the extension)
record_name = '100'

# Construct the absolute file path to the record file
file_path = f"{path}/{record_name}"

# Read the record using the 'rdrecord' function
record = wfdb.rdrecord(file_path)

# Access the signal data from the record object
signal = record.p_signal

# Access the metadata from the record object
metadata = record.__dict__

# Print the metadata of the record
print(metadata)