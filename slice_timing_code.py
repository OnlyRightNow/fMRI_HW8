""" Slice timing exercise
"""
#: Compatibility with Python 2
from __future__ import print_function  # print('me') instead of print 'me'
from __future__ import division  # 1/2 == 0.5, not 0

#: Import common modules
import numpy as np  # the Python array package

np.set_printoptions(precision=4, suppress=True)  # print to 4 DP
import matplotlib.pyplot as plt  # the Python plotting package
import nibabel as nib
from scipy.interpolate import InterpolatedUnivariateSpline

#: Set defaults for plotting
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

#: Load the image 'ds114_sub009_t2r1.nii' with nibabel
# Get the data array from the image
import nibabel as nib

img = nib.load('ds114_sub009_t2r1.nii')
data = img.get_data()
data.shape

#: Remove the first volume by slicing
fixed_data = data[..., 1:]
fixed_data.shape

# - Slice out time series for voxel (23, 19, 0)
slice0 = fixed_data[23, 19, 0, :]
# - Slice out time series for voxel (23, 19, 1)
slice1 = fixed_data[23, 19, 1, :]
# - Plot both these time series against volume number, on the same graph
plt.plot(slice0)
plt.plot(slice1)
plt.show()

#: The time between scans
TR = 2.5

# - Make time vector containing start times in second of each volume,
# - relative to start of first volume.
# - Call this `slice_0_times`
slice_0_times = np.arange(fixed_data.shape[-1]) * TR

# - Make time vector containing start times in seconds of z slice 1,
# - relative to start of first volume.
# - Call this `slice_1_times`
slice_1_times = slice_0_times + TR / 2
# - Plot first 10 values of slice 0 times against first 10 of slice 0
# - time series;
# - Plot first 10 values of slice 1 times against first 10 of slice 1
# - time series.
# - Use ':+' marker
plt.plot(slice_0_times[:10], slice0[:10], ':+')
plt.plot(slice_1_times[:10], slice1[:10], ':+')
plt.show()

# - Import `InterpolatedUnivariateSpline` from `scipy.interpolate`
# - Make a new linear (`k=1`) interpolation object for slice 1, with
# - slice 1 times and values.
inter = InterpolatedUnivariateSpline(slice_1_times, slice1, k=1)
# - Call interpolator with `slice_0_times` to get estimated values
slice1_est = inter(slice_0_times)
# - Plot first 10 values of slice 0 times against first 10 of slice 0
# - time series;
# - Plot first 10 values of slice 1 times against first 10 of slice 1
# - time series;
# - Plot first 10 values of slice 0 times against first 10 of
# - interpolated slice 1 time series.
plt.plot(slice_0_times[:10], slice0[:10], ':+')
plt.plot(slice_1_times[:10], slice1[:10], ':+')
plt.plot(slice_0_times[:10], slice1_est[:10], 'kx')
plt.show()
# - Copy old data to a new array
new_array = fixed_data.copy()
# - loop over all x coordinate values
# - loop over all y coordinate values
# - extract the time series at this x, y coordinate for slice 1;
# - make a linear interpolator object with the slice 1 times and the
# - extracted time series;
# - resample this interpolator at the slice 0 times;
# - put this new resampled time series into the new data at the same
# - position.
for i in range(fixed_data.shape[0]):
    for j in range(fixed_data.shape[1]):
        times = fixed_data[i, j, 1, :]
        inter = InterpolatedUnivariateSpline(slice_1_times, times, k=1)
        new_times = inter(slice_0_times)
        new_array[i, j, 1, :] = new_times
# - Make acquisition_order vector, length 30, with values:
# - 0, 15, 1, 16 ... 14, 29
acquisition_order = np.zeros(30)
ac_index = 0
for i in range(0, 30, 2):
    acquisition_order[i] = ac_index
    ac_index += 1
for i in range(1, 30, 2):
    acquisition_order[i] = ac_index
    ac_index += 1
# - Divide acquisition_order by number of slices, multiply by TR
acquisition_order_bla = acquisition_order/30 *TR
# - For each z coordinate (slice index):
# - # Make `slice_z_times` vector for this slice
# - ## For each x coordinate:
# - ### For each y coordinate:
# - #### extract the time series at this x, y, z coordinate;
# - #### make a linear interpolator object with the `slice_z_times` and
# -      the extracted time series;
# - #### resample this interpolator at the slice 0 times;
# - #### put this new resampled time series into the new data at the
# -      same position
for z in range(fixed_data.shape[2]):
    slice_z = slice_0_times + acquisition_order_bla[z]
    for x in range(fixed_data.shape[0]):
        for y in range(fixed_data.shape[1]):
            times = fixed_data[x, y, z, :]
            inter = InterpolatedUnivariateSpline(slice_z, times, k=1)
            new_times = inter(slice_0_times)
            new_array[x, y, z, :] = new_times
