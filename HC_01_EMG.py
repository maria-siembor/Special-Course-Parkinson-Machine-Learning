import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from HC_01_IMU import acc, pos, vel


#TA - Tibialis Anterior
#GA - Gastrocnemius
#HC_01 - Healthy Control 01
file_path_TA_HC_01 = 'C:/Users/s233183/OneDrive - Danmarks Tekniske Universitet/Desktop/Spring 2025/NRT-Lab SC/Data Collection/HC/HC_01/EMG/T_6/Tibialis_Anterior.txt'
file_path_GA_HC_01 = 'C:/Users/s233183/OneDrive - Danmarks Tekniske Universitet/Desktop/Spring 2025/NRT-Lab SC/Data Collection/HC/HC_01/EMG/T_6/Gastrocnemius.txt'

data_TA_HC_01 = pd.read_csv(file_path_TA_HC_01, comment='#', header=None, delim_whitespace=True, names=['timestamp', 'EMG'])
data_GA_HC_01 = pd.read_csv(file_path_GA_HC_01, comment='#', header=None, delim_whitespace=True, names=['timestamp', 'EMG'])

data_TA_HC_01['time_seconds'] = (data_TA_HC_01['timestamp'] - data_TA_HC_01['timestamp'].iloc[0]) / 1000.0
data_GA_HC_01['time_seconds'] = (data_GA_HC_01['timestamp'] - data_GA_HC_01['timestamp'].iloc[0]) / 1000.0

emg_signal_HC_01 = pd.merge_asof(
    data_TA_HC_01.sort_values('time_seconds'), 
    data_GA_HC_01.sort_values('time_seconds'), 
    on='time_seconds', 
    direction='nearest'
)

#emg_signal = data_TA_HC_01['EMG'].values  # Extract EMG as a NumPy array
fs = 1000  # Sampling rate (Hz) - adjust based on your data
nyquist = 0.5 * fs

plt.figure(figsize=(12, 4))
plt.plot(emg_signal_HC_01['time_seconds'], emg_signal_HC_01['EMG_x'], label='Tibialis Anterior')
plt.plot(emg_signal_HC_01['time_seconds'], emg_signal_HC_01['EMG_y'], label='Gastrocnemius')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Raw EMG Signal')
plt.grid()
plt.show()

acc = pd.DataFrame({
    'time_seconds': np.linspace(0, 10, 100),  # Example time data
    'some_other_column': np.random.rand(100)  # Example additional data
})

start_time = acc['time_seconds'].iloc[0]
end_time = acc['time_seconds'].iloc[-1]

emg_trimmed_HC_01 = emg_signal_HC_01[
    (emg_signal_HC_01['time_seconds'] >= start_time) &
    (emg_signal_HC_01['time_seconds'] <= end_time)
].copy()  # important to copy!

emg_trimmed_HC_01.reset_index(drop=True, inplace=True)

plt.figure(figsize=(12, 4)) 
plt.plot(emg_trimmed_HC_01['time_seconds'], emg_trimmed_HC_01['EMG_x'], label='Tibialis Anterior')
plt.plot(emg_trimmed_HC_01['time_seconds'], emg_trimmed_HC_01['EMG_y'], label='Gastrocnemius')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('Trimmed EMG Signal')
plt.grid()
plt.show()


low, high = 20 / nyquist, 450 / nyquist  # Adjusted cutoff frequencies to valid range
b, a = signal.butter(4, [low, high], btype='bandpass')

b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs)

b_low, a_low = signal.butter(4, 4 / nyquist, btype='lowpass')

for emg_col, prefix in [('EMG_x', 'TA'), ('EMG_y', 'GA')]:
    emg_trimmed_HC_01[f'{prefix}_zeromean'] = emg_trimmed_HC_01[emg_col] - np.mean(emg_trimmed_HC_01[emg_col])
    
    emg_trimmed_HC_01[f'{prefix}_filtered'] = signal.filtfilt(b, a, emg_trimmed_HC_01[f'{prefix}_zeromean'])
    
    emg_trimmed_HC_01[f'{prefix}_notch'] = signal.filtfilt(b_notch, a_notch, emg_trimmed_HC_01[f'{prefix}_filtered'])
    
for emg_col, prefix in [('EMG_x', 'TA'), ('EMG_y', 'GA')]:
    emg_trimmed_HC_01[f'{prefix}_rectified'] = np.abs(emg_trimmed_HC_01[f'{prefix}_notch'])
    
    emg_trimmed_HC_01[f'{prefix}_envelope'] = signal.filtfilt(b_low, a_low, emg_trimmed_HC_01[f'{prefix}_rectified'])


plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
plt.plot(emg_signal_HC_01['time_seconds'], emg_signal_HC_01['EMG_x'], label='Tibialis Anterior (Raw)', color='blue')
plt.plot(emg_signal_HC_01['time_seconds'], emg_signal_HC_01['EMG_y'], label='Gastrocnemius (Raw)', color='green')
plt.title('Raw EMG Signals')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(emg_trimmed_HC_01['time_seconds'], emg_trimmed_HC_01['TA_zeromean'], label='TA (Zero-mean)', color='lightblue')
plt.plot(emg_trimmed_HC_01['time_seconds'], emg_trimmed_HC_01['GA_zeromean'], label='GA (Zero-mean)', color='lightgreen')
plt.title('After DC Offset Removal')
plt.legend()
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(emg_trimmed_HC_01['time_seconds'], emg_trimmed_HC_01['TA_filtered'], label='TA Filtered', color='red')
plt.plot(emg_trimmed_HC_01['time_seconds'], emg_trimmed_HC_01['GA_filtered'], label='GA Filtered', color='purple')
plt.title('Bandpass Filtered (20-450 Hz)')
plt.legend()
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(emg_trimmed_HC_01['time_seconds'], emg_trimmed_HC_01['TA_envelope'], label='TA Envelope', color='orange')
plt.plot(emg_trimmed_HC_01['time_seconds'], emg_trimmed_HC_01['GA_envelope'], label='GA Envelope', color='brown')
plt.title('Activation Envelopes (4 Hz Lowpass)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
