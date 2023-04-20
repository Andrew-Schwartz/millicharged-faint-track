import math
import numpy as np
import skimage
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import cuda
from numba.types import int32
import time
from skimage.feature.peak import peak_local_max

# clear the memory - help if too many threads/blocks are used
cuda.devices.reset()

nwires = 400  # Number of TPC wires (dimension along the beam)
nticks = 1600  # Number of TPC time ticks (dimension transverse to the beam)
noise_mu = 0  # Noise amplitude [# electrons]
noise_std = 250  # Noise standard deviation [# electrons]
signal_q = 0.075  # Signal charge [units of e]

# Random noise and an empty signal to start with
event = np.random.normal(noise_mu, noise_std, size=nticks * nwires).reshape((nticks, nwires))
signal = np.zeros_like(event)
print(event.shape)

# Lines approximating track(s)
ntracks = 2
truth = np.empty(shape=(2, ntracks), dtype=float)

# Very approximate scale factor for converting signal amplitude to # electrons
sf = 1e6 / 23.6 / 10 * signal_q ** 2
print("sf = ", sf)
print("2.1 * sf = ", 2.1 * sf)
print("0.15 * sf = ", 0.15 * sf)

for i in range(ntracks):
    start_y, end_y = np.random.randint(0, nticks - 1, size=2)  # Start/end points
    rr, cc, val = skimage.draw.line_aa(start_y - 1, 0, end_y - 1, nwires - 1)  # Fuzzy line
    line = event[rr, cc]  # All pixels along the line

    signal[rr, cc] += scipy.stats.moyal.rvs(size=len(line), loc=2.1 * sf, scale=0.15 * sf)  # Signal with fluctuations
    truth[i] = [start_y, end_y]  # Keep track of the true start/ends

    event += signal  # Add this track to the background noise


# fig, ax = plt.subplots()
# im = ax.imshow(event, interpolation='none', cmap=cm.gray)
# ax.set_xlabel('Wire')
# ax.set_ylabel('Tick')
# plt.colorbar(im)
# plt.tight_layout()
# plt.savefig('fake_mcp.png', dpi=300)
# plt.show()

@cuda.jit(device=True)
def sum_along_line(x0, y0, x1, y1, event):
    sum = 0
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    ed = 1 if dx + dy == 0 else math.sqrt(dx ** 2 + dy ** 2)
    while True:
        # img[x0, y0] = 255 * (1 - abs(err - dx + dy) / ed)
        sum += event[x0, y0]
        e2 = err
        x2 = x0
        if 2 * e2 >= -dx:
            if x0 == x1:
                break
            if e2 + dy < ed:
                # img[x0, y0 + sy] = 255 * (1 - (e2 + dy) / ed)
                sum += event[x0, y0 + sy]
            err -= dy
            x0 += sx
        if 2 * e2 <= dy:
            if y0 == y1:
                break
            if dx - e2 < ed:
                # img[x2 + sx, y0] = 255 * (1 - (dx - e2) / ed)
                sum += event[x2 + sy, y0]
            err += dx
            y0 += sy
    return sum


@cuda.jit
def make_tracks(nticks, event, sums):
    idx1, idx2 = cuda.grid(2)
    if idx1 < nticks and idx2 < nticks:
        sums[idx1, idx2] = sum_along_line(idx1, 0, idx2, nwires - 1, event)


start_gpu = time.time()

# d_images = cuda.to_device([[gen_img() for i2 in range(0, nticks)] for i1 in range(0, nticks)])
d_sums = cuda.to_device(np.zeros((nticks, nticks)))
d_event = cuda.to_device(event)
make_tracks[(32, 32), (32, 32)](nticks, d_event, d_sums)
sums = d_sums.copy_to_host()

elapsed_gpu = time.time() - start_gpu
print("gpu time =", elapsed_gpu)

fig, ax = plt.subplots(1, 2, figsize=(10,5))

im0 = ax[0].imshow(sums.T)
plt.colorbar(im0, shrink=0.6, ax=ax[0])

print('True start/end points:')
print(truth)

peaks2 = peak_local_max(sums, min_distance=1, num_peaks=ntracks)
ax[0].plot(*list(peaks2.T), 'o', c='red', alpha=0.75, label='Peak finder')
print('Found peaks (local max):')
print(peaks2)

ax[0].set_xlabel('Start $y$')
ax[0].set_ylabel('End $y$')
ax[0].legend(loc='lower right')

### Draw the true and found lines
im1 = ax[1].imshow(event, interpolation='none', cmap=cm.gray)

plt.colorbar(im1, shrink=0.6, ax=ax[1])

# Truth, shifted vertically a bit
for i, (s, e) in enumerate(truth):
    ax[1].axline((0, s), (nwires-1, e), ls='--', c='blue', lw=2, label='Truth')

# Found, shifted vertically a bit
for i, (s, e) in enumerate(peaks2):
    ax[1].axline((0, s), (nwires-1, e), ls='--', c='red', lw=2, label='Peak finder')

ax[1].set_xlabel('Wire')
ax[1].set_ylabel('Tick')
ax[1].set_xlim(0, nwires)
ax[1].set_ylim(0, nticks)

plt.tight_layout()
plt.savefig('trk_mcp.png', dpi=300)
plt.show()

#%%
