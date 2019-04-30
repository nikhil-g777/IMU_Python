import csv
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import scipy.fftpack
from scipy.signal import butter, lfilter, find_peaks

measurements_z = []
accelerometer_readings_z = []
line_count = 0
offset = 100
fs = 1000./20.
alpha = 0.7
with open('accelerometer.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
            reading = float(row["accelerometer_z"])
            measurements_z.append(reading)
            accelerometer_readings_z.append(np.array([[reading], [reading]]))
        line_count += 1
    print(f'Processed {line_count} lines.')


filtered_outputs = []
my_filter = KalmanFilter(dim_x=2, dim_z=2)
my_filter.F = np.array([[1,0], [0,1]])    # state transition matrix
my_filter.x = accelerometer_readings_z[0]       # initial state (location and velocity)
my_filter.H = np.array([[1,0], [0,1]])    # Measurement function
my_filter.P *= 100.                 # covariance matrix
my_filter.R = 70.                    # state uncertainty
my_filter.Q = Q_discrete_white_noise(2, 1., 0.1) # process uncertainty

counter = 0
filtered_accelerometer_z = []
lowpass_accelerometer_z = []
lowpass_measured_z = []
predicted_accelerometer_z = []
filtered_outputs = []
counters = []

######################### Kalman Filtering#############################
                    
while counter < len(accelerometer_readings_z):
    my_filter.predict()
    predicted_accelerometer_z.append(my_filter.x[0][0])
    my_filter.update(accelerometer_readings_z[counter])
    x = my_filter.x
    filtered_accelerometer_z.append(x[0][0])
    filtered_outputs.append(x)
    counters.append(counter)
    counter += 1

counters = counters[offset:]
# filtered_accelerometer_z = filtered_accelerometer_z[offset:]
# measurements_z = measurements_z[offset:]
########################      Low Pass Filter #########################
x = 0
for reading in measurements_z:
        x = alpha * x + (1-alpha) * reading
        reading = reading - x
        lowpass_measured_z.append(reading)
lowpass_measured_z = lowpass_measured_z[offset:]
#######################################################################

########################      Low Pass Filter #########################
x = 0
for reading in filtered_accelerometer_z:
        x = alpha * x + (1-alpha) * reading
        reading = reading - x
        lowpass_accelerometer_z.append(reading)
lowpass_accelerometer_z = lowpass_accelerometer_z[offset:]
#######################################################################

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# y = butter_bandpass_filter(lowpass_measured_z, 0.1, 0.32, fs, 1)
# fig1, ax1 = plt.subplots()
# ax1.plot(counters, y)
# plt.show()
#######################       FFT        #############################
# N = 2999 - offset
# T = 20./1000.
# x = np.linspace(0.0, N*T, N)
# xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

# fig1, ax1 = plt.subplots()
# yf = scipy.fftpack.fft(lowpass_accelerometer_z)
# ax1.plot(xf, 2.0/N * np.abs(yf[:N//2]))
# plt.show()
######################################################################

#######################   peak finding   #############################
# peaks, _ = find_peaks(lowpass_accelerometer_z, width=4)
# print(peaks)
# peak_values = []
# for peak in peaks:
#     peak_values.append(lowpass_accelerometer_z[peak])
# plt.plot(peaks, peak_values, "ob")
######################################################################


#######################   threshold detector   #######################
threshold_readings = []
threshold_indexes = []
threshold = 0.07
width_threshold = 15
recording = False
startIndex = 0
currentIndex = 0
minimum_duration = 10
recordings = []
for index, sensor_reading in enumerate(lowpass_measured_z):
    index += offset
    if sensor_reading > threshold:
        if not recording:
            recording = True
            startIndex = index
            currentIndex = index
        else:
            threshold_readings.append(sensor_reading)
            threshold_indexes.append(index)
            currentIndex = index
    else:
        if recording:
            if (index - currentIndex) > width_threshold:
                recording = False
                if index-startIndex > minimum_duration:
                    recordings.append((index - startIndex)*19)

print(recordings)
plt.plot(threshold_indexes, threshold_readings, "ob")

######################################################################


plt.plot(counters, lowpass_measured_z, color="red")
plt.plot(counters, lowpass_accelerometer_z, color="green")

plt.xlabel('time')
plt.ylabel('Z-axis accelerometer reading')
plt.show()
# plt.savefig("one.jpg")
