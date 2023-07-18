import os

# select which gpu to use
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

import math
import numpy as np
from numba import cuda, vectorize, NumbaPerformanceWarning, njit
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import warnings
import cupy as cp
from cucim.skimage.feature import peak_local_max
from time import perf_counter_ns

import gpu_random

# Filter out the NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Constants
nwires = 1664  # Number of TPC wires (dimension along the beam)
nticks = 3400  # Number of TPC time ticks (dimension transverse to the beam)

# geometry info for drawing lines back to target
target = 11000  # Distance to target, cm
dwire = 0.3  # Distance between wires, cm/wire
dtick = 0.05  # Distance between ticks, cm/tick
length_wires = nwires * dwire  # 449.2 cm
length_ticks = nticks * dtick  # 170 cm
# Target is a cylinder, 71.1 cm long with radius 0.51 cm, see section II B of
# https://journals-aps-org.proxy.libraries.rutgers.edu/prd/pdf/10.1103/PhysRevD.79.072002
# We use slightly larger radius to account for various inaccuracies.
# todo revisit these increases
y_min = 82  # Lower y limit, cm
y_max = 88  # Upper y limit, cm
length_target = 74  # cm


# calculation of the maximum distance between the points on the front side for any point on the back side
# https://www.desmos.com/calculator/gncrncjxhi

@njit(nogil=True)
def intercept(y_end: float, back: bool, y: float) -> float:
    """
    For a given height (cm) on far side of the detector, calculate the point on the
    close side that is on a line between the far point and one of four boundary points
    of the target.
    :param y_end: the y coordinate of the end point, `f` in Desmos
    :param back: true if the point is the back of the target, false for the front point, `b` Desmos
    :param y: y_min or y_max, `i` in Desmos
    """
    back_length = length_target if back else 0
    slope = (y_end - y) / (target + length_wires + back_length)
    return slope * (target + back_length) + y


@cuda.jit
def intercept_kern(y_end, back, y, out):
    idx = cuda.grid(1)
    if idx < len(y_end):
        out[idx] = intercept(y_end[idx], back[idx], y)


def delta_ticks(y_end: float) -> float:
    max_intercept = intercept(y_end, y_end > y_max, y_max)
    min_intercept = intercept(y_end, y_end < y_min, y_min)
    delta_ticks_cm = max_intercept - min_intercept
    return math.ceil(delta_ticks_cm / dtick)


# the largest separation will either be at the very top or very bottom tick,
# or a point between y_min and y_max (all such points have same separation)
tick_separation = max(delta_ticks(y_end) for y_end in [0, y_min + 1, length_ticks])

# if tick_separation <= 1024, can fit each row into one block
# if tick_separation <= ~25, can fit entire track sum on gpu
print(f'tick_separation = {tick_separation}')


@cuda.jit
def get_start_end_pairs_kern(rng_states, pairs):
    idx = cuda.grid(1)
    if idx < pairs.shape[0]:
        y1 = gpu_random.uniform(rng_states, idx, 0, nticks)
        y1_cm = y1 * dtick
        y0_min_cm = intercept(y1_cm, y1_cm < y_min, y_min)
        y0_max_cm = intercept(y1_cm, y1_cm > y_max, y_max)
        y0_min = math.floor(y0_min_cm / dtick)
        y0_max = math.ceil(y0_max_cm / dtick)
        y0 = gpu_random.uniform(rng_states, idx, y0_min, y0_max)
        pairs[idx, 0] = y0
        pairs[idx, 1] = y1


def get_start_end_pairs(n_pairs, rng_states):
    pairs = cp.empty((n_pairs, 2))
    get_start_end_pairs_kern.forall(n_pairs)(rng_states, pairs)
    return pairs


# taken from `standard_detsim_sbnd.fcl`
noise_function_parameters = [
    1.19777e1,
    170000.0,
    4.93692e3,
    1.03438e3,
    2.33306e2,
    1.36605,
    4.08741,
    3.5e-3,
    9596.0
]

# from print out sbndcode values
wirelength = 400
ntick = 4096
clock_freq = 1e9
samplerate = 1000 * (1 / clock_freq)

# from SBNDuBooNEDataDrivenNoiseService_service.cc
noise_function_parameters[6] = 0.395 + 0.001304 * wirelength
nfp0, nfp1, nfp2, nfp3, nfp4, nfp5, nfp6, nfp7, nfp8 = noise_function_parameters
binwidth = 1 / (ntick * samplerate * 1e-6)

DL = 3.74  # cm²/s (longitudinal electron diffusion coeff)
DL = DL * 1e-6  # cm²/μs (longitudinal electron diffusion coeff)
v_d = 1.076  # mm/μs (drift velocity)
v_d = v_d * 1e-1  # cm/μs (drift velocity)
sigma_t0_2 = 1.98  # μs² (time width)
E = 273.9  # V/cm (E field)

# from https://lar.bnl.gov/properties/
energy_per_electron = 23.6  # eV/e⁻ pair
# from https://arxiv.org/pdf/1802.08709.pdf or https://arxiv.org/pdf/1804.02583.pdf
electrons_per_adc = 187  # e⁻/ADC
survival_probability = 0.6980  # % (survive recombination)
dEdx_mean = 2.1173  # MeV/cm
# dEdx_mean = dEdx_mean * 1e6 # eV/cm
z = 1  # % (mCP charge fraction)
dEdx_mean *= z ** 2  # scale based on charge
# print("dEdx_mean = ", dEdx_mean)

width = 200  # cm (x)
height = 400  # cm (y)
length = 500  # cm (beam direction, z)
n_steps = nwires  # number

# magic number to get shape of landau distribution right
width_factor = 10
# cutoff energy (eV)
energy_loss_cutoff = 1e5


@cp.fuse
def pfn_f1(x):
    """
    from SBNDuBooNEDataDrivenNoiseService_service.cc
    """
    term1 = nfp0 * 1 / (x / 1000 * nfp8 / 2)
    term2 = nfp1 * cp.power(cp.e, (-0.5 * cp.power((((x / 1000 * nfp8 / 2) - nfp2) / nfp3), 2)))
    term3 = cp.power(cp.e, (-0.5 * cp.power(x / 1000 * nfp8 / (2 * nfp4), nfp5)))
    return (term1 + (term2 * term3) * nfp6) + nfp7


@cp.fuse
def calc_noise_frequency_along_wire(tick, poisson_number, phase):
    """
    from SBNDuBooNEDataDrivenNoiseService_service.cc
    """
    pfnf1val = pfn_f1((tick + 0.5) * binwidth)
    pval = pfnf1val * poisson_number
    return pval * cp.cos(phase) + 1j * pval * cp.sin(phase)


def gen_noise():
    """Generate noise for an event.

    This is the starting point of the simulation.

    Returns a 2d cp.ndarray
    """

    poisson_mu = 3.30762
    # generate the poisson random numbers
    poisson_random = cp.random.poisson(poisson_mu, nwires * nticks, dtype=np.float32) / poisson_mu
    # cp.random.poisson can only generate 1d arrays, so reshape it to be 2d for ease of access
    poisson_random = cp.reshape(poisson_random, (nwires, nticks))

    # generate the random phase
    uniform_random = cp.random.uniform(0, 2 * math.pi, (nwires, nticks))

    # generate tick number for each wire
    ticks = cp.tile(cp.arange(nticks, dtype=np.int16), (nwires, 1))

    # calculate frequencies
    noise_d = calc_noise_frequency_along_wire(ticks, poisson_random, uniform_random)
    # kinda magic constant, but basically chosen to take the range to ±6 ADCs
    # todo figure out what the conversion should actually be
    noise_d *= 25000

    # inverse fourier transform each wire to get the noise
    noise_d = cp.array([cp.real(cp.fft.ifft(channel_freqs)) for channel_freqs in noise_d])
    return noise_d


@cuda.jit
def shuffle_noise_kern(noise, new_noise, indices, roll_amounts):
    w, t = cuda.grid(2)
    if w < nwires and t < nticks:
        new_w = indices[w]
        new_t = (t + roll_amounts[w]) % nticks
        new_noise[new_w, new_t] = noise[w, t]


def shuffle_noise(noise):
    new_noise = cp.empty_like(noise)
    # wire at noise[0] goes into new_noise[indices[0]]
    indices = cp.arange(0, nwires)
    cp.random.shuffle(indices)
    roll_amounts = cp.random.randint(0, nwires, size=nwires)
    threads = (16, 32)
    blocks = tuple(math.ceil(noise.shape[axis] / threads[axis]) for axis in [0, 1])
    shuffle_noise_kern[blocks, threads](noise, new_noise, indices, roll_amounts)
    return new_noise


@vectorize(['int32(float32)'], target='cuda')
def get_n_ionization_electrons(energy_loss):
    # convert to eV, get rid of negatives, and put in an upper threshold (otherwise, the
    # long tail of the landau makes some super huge values wash everything else out)
    # todo eventually ask Mastbaum if this 1e5 cutoff makes sense
    energy_loss = min(max(1e6 * energy_loss, 0), energy_loss_cutoff)
    # number of elections ionized
    n_electrons = energy_loss / energy_per_electron
    # number of electrons that make it to readout
    return int(n_electrons * survival_probability)


MAX_IONIZATION_ELECTRONS = int(energy_loss_cutoff / energy_per_electron * survival_probability)


@cp.fuse
def get_sigma_x(x):
    # drift time, μs
    t = x / v_d
    # equation 3.1 of https://iopscience.iop.org/article/10.1088/1748-0221/16/09/P09025/pdf
    # this is the width on the collection plane (μs²)
    sigma_t = cp.sqrt(sigma_t0_2 + (2 * DL / v_d ** 2) * t)
    return sigma_t * v_d * 6  # 6 is a fudge factor


@cuda.jit
def get_adcs(rngs, y0, y_slope, ionization_electrons, sigma_xs, adcs):
    z = cuda.grid(1)
    if z < n_steps:
        y = y0 + y_slope * z
        sigma_x = sigma_xs[z]
        ionization_electrons = ionization_electrons[z]
        for i in range(ionization_electrons):
            # wire, gaussian distributed around the current wire (z)
            wire = sigma_x * xoroshiro128p_normal_float32(rngs, z)
            wire += z
            wire = int(wire)

            # tick, gaussian distributed around the current tick (y) (this is kinda weird?)
            tick = sigma_x * xoroshiro128p_normal_float32(rngs, z)
            # tick = tick / nticks * height
            tick += y
            # tick = tick / height * nticks
            tick = int(tick)

            if 0 <= wire < nwires and 0 <= tick < nticks:
                adcs[z][i][0] = wire
                adcs[z][i][1] = tick


CHUNKS = 128
print(f'CHUNKS = {CHUNKS}')


@cuda.jit
def add_to_noise(adcs, noise_partial_reduction):
    idx = cuda.grid(1)

    if idx < CHUNKS:
        for chunk in range(idx, len(adcs), CHUNKS):
            for i in range(MAX_IONIZATION_ELECTRONS):
                wire = adcs[chunk][i][0]
                tick = adcs[chunk][i][1]
                if 0 <= wire < nwires and 0 <= tick < nticks:
                    noise_partial_reduction[idx][wire][tick] += 1


def simulate_track(event, x0, y0, x1, y1, rng_states):
    """Simulates a mCP crossing the detector, from (x0, y0) to (x1, y1)

    Modifies `noise` in place.
    """

    # random energy losses
    energy_losses_d = gpu_random.landau_rvs(n_steps, dEdx_mean, dEdx_mean / width_factor, rng_states)

    # get # of ionization electrons
    ionization_electrons_d = get_n_ionization_electrons(energy_losses_d)

    # get sigma of normal distribution
    x_slope = (x1 - x0) / n_steps
    # distance to readout plane, cm
    xs_d = x0 + cp.arange(n_steps, dtype=np.float32) * x_slope
    sigma_xs_d = get_sigma_x(xs_d)

    # get individual
    y_slope = (y1 - y0) / n_steps
    adcs_d = cp.full((n_steps, MAX_IONIZATION_ELECTRONS, 2), -1, dtype=np.int32)
    get_adcs.forall(n_steps)(rng_states, y0, y_slope, ionization_electrons_d, sigma_xs_d, adcs_d)
    # adcs /= electrons_per_adc

    # reduce
    noise_partial_reduction_d = cp.zeros((CHUNKS, nwires, nticks), np.float32)
    add_to_noise.forall(CHUNKS)(adcs_d, noise_partial_reduction_d)

    reduced = noise_partial_reduction_d.sum(axis=0)
    reduced /= electrons_per_adc
    event += reduced


# implementation from paper cited in `skimage.draw.line_aa`:
# Listing 16 of http://members.chello.at/easyfilter/Bresenham.pdf
@cuda.jit(device=True)
def sum_along_line(x0, y0, x1, y1, event, aa, normalize):
    total = 0
    npoints = 0
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    ed = 1 if dx + dy == 0 else math.sqrt(dx ** 2 + dy ** 2)
    while True:
        # img[x0, y0] = 255 * (1 - abs(err - dx + dy) / ed)
        # 1 if aa is False, if aa is True (= 1) then subtract the antialiasing amount (multiplication should be
        # faster than branching esp on GPU)
        aa_mult = 1 - aa * abs(err - dx + dy) / ed
        total += aa_mult * event[x0, y0]
        npoints += aa_mult
        e2 = err
        x2 = x0
        if 2 * e2 >= -dx:
            if x0 == x1:
                break
            if e2 + dy < ed:
                # img[x0, y0 + sy] = 255 * (1 - (e2 + dy) / ed)
                aa_mult = 1 - aa * (e2 + dy) / ed
                total += aa_mult * event[x0, y0 + sy]
                npoints += aa_mult
            err -= dy
            x0 += sx
        if 2 * e2 <= dy:
            if y0 == y1:
                break
            if dx - e2 < ed:
                # img[x2 + sx, y0] = 255 * (1 - (dx - e2) / ed)
                aa_mult = 1 - aa * (dx - e2) / ed
                total += aa_mult * event[x2 + sy, y0]
                npoints += aa_mult
            err += dx
            y0 += sy
    return total / npoints if normalize else total


# 1024 is the max # of threads per block
threads = 1024
# how many end tick values each thread is able to calculate
ticks_per_block = threads // tick_separation
# number of blocks
blocks = math.ceil(nticks / ticks_per_block)

y_min_t = y_min / dtick


@cuda.jit
def make_tracks_kern(event, sums, aa, normalize):
    # cuda.grid(1) == cuda.threadIdx + cuda.blockIdx.x * cuda.blockDim.x
    tid = cuda.threadIdx.x
    if tid < ticks_per_block * tick_separation:
        end = tid // tick_separation + cuda.blockIdx.x * ticks_per_block
        start = y_min_t + (abs(target) / (abs(target) + length_wires)) * (end - y_min_t)
        start = round(start) + tid % tick_separation
        if start < nticks:
            sums[start, end] = sum_along_line(0, start, nwires - 1, end, event, aa, normalize)


def track_sums(event):
    sums_d = cp.full((nticks, nticks), -1, np.float32)
    make_tracks_kern[blocks, threads](event, sums_d, True, True)
    return sums_d


np_rng = np.random.default_rng()
track_seed, pairs_seed = np_rng.integers(np.iinfo(np.uint64).max, size=2, dtype=np.uint64)

track_rng = create_xoroshiro128p_states(n_steps, seed=track_seed)
iters = 100
timing  = np.empty((5, iters))
n_times_noise_used = 10

time_all = True

# # Generating 50 million pairs takes about 0.25 seconds, so it will easily be done by the time that
# # all the initial ~10k pairs are fully simulated. Can't go much higher because of memory limits.
# n_pairs = 100
# with cuda.gpus[1]:
#     print('gpu1:', cuda.current_context().device.uuid)
#     pairs_rng = create_xoroshiro128p_states(n_pairs, seed=pairs_seed)
#     pairs = get_start_end_pairs(10, pairs_rng).get()

# found_peaks = []
# while len(found_peaks) < 100:
#     print(f'len(pairs) = {len(pairs)}')
#     print(f'len(found_peaks) = {len(found_peaks)}')
#     with cuda.gpus[1]:
#         print('gpu1:', cuda.current_context().device.uuid)
#         pairs_async = get_start_end_pairs(n_pairs, pairs_rng)
#
#     print('default gpu:', cuda.current_context().device.uuid)
#     for y0, y1 in pairs:
#         event = gen_noise()
#         simulate_track(event, 0, y0, n_steps - 1, y1, track_rng)
#         sums = track_sums(event)
#         peaks = peak_local_max(sums, min_distance=1, num_peaks=1)
#         found_peaks.append(peaks.get())
#     print(f'len(found_peaks) = {len(found_peaks)}')
#     pairs = pairs_async.get()
#
# print(found_peaks)

# compile all kernels
y0, y1 = np.array([2021, 2109], dtype=np.float32)
event = gen_noise()
shuffle_noise(event)
simulate_track(event, 0, y0, n_steps - 1, y1, track_rng)
sums = track_sums(event)
peak_local_max(sums, min_distance=1, num_peaks=1)

print(f'Timing {"individual parts, " if time_all else ""}{iters} iterations')

stored_noise = None
for i in range(iters):
    start = perf_counter_ns()

    if i % n_times_noise_used == 0:
        event = gen_noise()
    else:
        event = shuffle_noise(stored_noise)
    if time_all:
        cuda.synchronize()
    stop_gen_noise = perf_counter_ns()

    simulate_track(event, 0, y0, n_steps - 1, y1, track_rng)
    if time_all:
        cuda.synchronize()
    stop_sim_track = perf_counter_ns()

    sums = track_sums(event)
    if time_all:
        cuda.synchronize()
    stop_track_sums = perf_counter_ns()

    peaks = peak_local_max(sums, min_distance=1, num_peaks=1)
    if time_all:
        cuda.synchronize()
    stop_plm = perf_counter_ns()
    cuda.synchronize()

    stop = perf_counter_ns()
    if i % n_times_noise_used == 0:
        stored_noise = event

    # print(peaks)
    timing[0, i] = stop - start
    timing[1, i] = stop_gen_noise - start
    timing[2, i] = stop_sim_track - stop_gen_noise
    timing[3, i] = stop_track_sums - stop_sim_track
    timing[4, i] = stop_plm - stop_track_sums

timing *= 1e-6

print(f"              Elapsed time: {timing[0].mean():.3f} ± {timing[0].std():.3f} ms")
if time_all:
    print(f"  Elapsed time (gen_noise): {timing[1].mean():.3f} ± {timing[1].std():.3f} ms")
    print(f"  Elapsed time (sim_track): {timing[2].mean():.3f} ± {timing[2].std():.3f} ms")
    print(f"  Elapsed time (track_sum): {timing[3].mean():.3f} ± {timing[3].std():.3f} ms")
    print(f"        Elapsed time (plm): {timing[4].mean():.3f} ± {timing[4].std():.3f} ms")
