import argparse
import os

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('millicharged particle faint tracks')
parser.add_argument('chunks', type=int, help='how many chunks to use for `simulate_tracks`')
parser.add_argument('threads', type=int, help='how many threads to use for `track_sum`, 1024 is the max', default=32)
parser.add_argument('--iters', type=int, help='number of iterations to run', default=100)
parser.add_argument('-z', type=float, help='charge of mCP', default=1.0)
parser.add_argument('--mode', choices=['bench', 'profile', 'chunks', 'threads', 'run'], default='bench',
                    help='mostly used to determine what to print')
parser.add_argument('--no-time-all', action='store_false')
parser.add_argument('--gpu', type=int, default=3, required=False, help='which GPU to use, 0-3 on spsfarm')
args = parser.parse_args()
mode = args.mode
iters = args.iters

# select which gpu to use (this has to be done before any gpu functions are imported)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from time import perf_counter_ns
import math
import numpy as np
from numba import cuda, NumbaPerformanceWarning, njit
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import warnings
import cupy as cp
from cupy.random import RandomState
from cucim.skimage.feature import peak_local_max

import gpu_random

# Filter out the NumbaPerformanceWarning (it warns for small grid size, but that can't be avoided in some cases here)
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

# nopython jit so that it can be used on CPU & as a device function
@njit(nogil=True)
def intercept(y_end: cp.float32, back: bool, y: cp.float32) -> cp.float32:
    """
    For a given height (cm) on far side of the detector, calculate the point on the close side that is on a line between
    the far point and one of four boundary points of the target.
    This function is called l in the Desmos: https://www.desmos.com/calculator/gncrncjxhi.
    :param y_end: the y coordinate of the end point in cm, `f` in Desmos
    :param back: true if the point is the back of the target, false for the front point, `b` Desmos
    :param y: y_min or y_max in cm, `i` in Desmos
    """
    back_length = length_target if back else 0
    slope = (y_end - y) / (target + length_wires + back_length)
    return slope * (target + back_length) + y


@cuda.jit(fastmath=True)
def intercept_kern(y_end: cp.ndarray, back: cp.ndarray, y: cp.float32, out: cp.ndarray):
    """
    Calculate the intercept for many `y_end`s at once.
    """
    idx = cuda.grid(1)
    if idx < len(y_end):
        out[idx] = intercept(y_end[idx], back[idx], y)


def delta_ticks(y_end: cp.float32) -> int:
    """
    For a given height (cm) on the far side of the detector, calculate how many ticks are between the points on the
    close side of the detector on lines from that far side point to the top/bottom of the target.
    This function is the last equation in the Desmos, l(...1)-l(...0): https://www.desmos.com/calculator/gncrncjxhi.
    :param y_end: the y coordinate of the end point in cm, `f` in Desmos
    """
    max_intercept = intercept(y_end, y_end > y_max, y_max)
    min_intercept = intercept(y_end, y_end < y_min, y_min)
    delta_ticks_cm = max_intercept - min_intercept
    return math.ceil(delta_ticks_cm / dtick)


# the largest separation will either be at the very top or very bottom tick,
# or a point between y_min and y_max (all such points have same separation)
tick_separation = max(delta_ticks(y_end) for y_end in [0, y_min + 1, length_ticks])


# if tick_separation <= 1024, can fit each row into one block
# if tick_separation <= ~25, can fit entire track sum on gpu
# print(f'tick_separation = {tick_separation}')


@cuda.jit(fastmath=True)
def get_start_end_pairs_kern(rng_states: DeviceNDArray, pairs: cp.ndarray):
    """Populate `pairs` with pairs of (start, end) ticks that form a line to the target.

    Important: `len(rng_states) >= len(pairs)`
    """
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


def get_start_end_pairs(n_pairs: int, rng_states: DeviceNDArray):
    """Get `n_pairs` pairs of (start, end) ticks on a line pointing to the target.

    :return: A 2d cp.ndarray with shape (npairs, 2).
    """
    pairs = cp.empty((n_pairs, 2))
    # is a custom kernel because doing a python loop over each wire with cp.roll was slow
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
nfp0, nfp1, nfp2, nfp3, nfp4, nfp5, nfp6, nfp7, nfp8 = np.array(noise_function_parameters, dtype=cp.float32)
binwidth = cp.float32(1 / (ntick * samplerate * 1e-6))

DL = 3.74  # cm²/s (longitudinal electron diffusion coeff)
DL = DL * 1e-6  # cm²/μs (longitudinal electron diffusion coeff)
v_d = 1.076  # mm/μs (drift velocity)
v_d = v_d * 1e-1  # cm/μs (drift velocity)
sigma_t0_2 = 1.98  # μs² (time width)
E = 273.9  # V/cm (E field)

width = 200  # cm (x)
height = 400  # cm (y)
length = 500  # cm (beam direction, z)
n_steps = nwires  # number, should be at least nwires
step_length = dwire * nwires / n_steps  # cm

# from https://lar.bnl.gov/properties/
energy_per_electron = 23.6  # eV/e⁻ pair
# from https://arxiv.org/pdf/1802.08709.pdf or https://arxiv.org/pdf/1804.02583.pdf
electrons_per_adc = 187  # e⁻/ADC
survival_probability = 0.6980  # % (survive recombination)
dEdx_mean = 2.1173  # MeV/cm
# dEdx_mean = dEdx_mean * 1e6 # eV/cm
dE_mean = dEdx_mean * step_length
z = args.z  # % (mCP charge fraction)
dE_mean *= z ** 2  # scale based on charge
dE_mean = cp.float32(dE_mean)
# print("dEdx_mean = ", dEdx_mean)

# magic number to get shape of landau distribution right
width_factor = 10
# cutoff energy (eV)
energy_loss_cutoff = 1e5


@cp.fuse
def pfn_f1(x: cp.float32) -> cp.float32:
    """
    from SBNDuBooNEDataDrivenNoiseService_service.cc
    """
    term1 = nfp0 * cp.float32(1) / (x / cp.float32(1000) * nfp8 / cp.float32(2))
    term2 = nfp1 * cp.power(cp.float32(cp.e), (
            -cp.float32(0.5) * cp.power((((x / cp.float32(1000) * nfp8 / cp.float32(2)) - nfp2) / nfp3),
                                        cp.float32(2))))
    term3 = cp.power(cp.float32(cp.e),
                     (-cp.float32(0.5) * cp.power(x / cp.float32(1000) * nfp8 / (cp.float32(2) * nfp4), nfp5)))
    return (term1 + (term2 * term3) * nfp6) + nfp7


@cp.fuse
def calc_noise_frequency_along_wire(tick: cp.float32, poisson_number: cp.float32, phase: cp.float32) -> cp.complex64:
    """
    from SBNDuBooNEDataDrivenNoiseService_service.cc
    """
    pfnf1val = pfn_f1((tick + cp.float32(0.5)) * binwidth).astype(cp.float32)
    pval = pfnf1val * poisson_number
    real = pval * cp.cos(phase)
    imag = pval * cp.sin(phase)
    return real.astype(cp.float32) + 1j * imag.astype(cp.float32)


def gen_noise(rng: RandomState):
    """Generate noise for an event.

    This is the starting point of the simulation.

    :return: A 2d cp.ndarray with shape (nwires, nticks)
    """
    # from sbndcode
    poisson_mu = 3.30762
    # generate the poisson random numbers
    poisson_random = rng.poisson(poisson_mu, size=(nwires, nticks), dtype=np.int32) / poisson_mu

    # generate the random phase
    uniform_random = rng.uniform(0, 2 * math.pi, size=(nwires, nticks), dtype=cp.float32)
    # generate tick number for each wire: (0, 1, 2, 3, 4, 5, ..., nticks - 1) copied nwires times
    ticks = cp.tile(cp.arange(nticks, dtype=np.int32), (nwires, 1))

    # calculate frequencies
    noise_d = calc_noise_frequency_along_wire(ticks, poisson_random, uniform_random)
    # todo figure out what the conversion should actually be
    noise_d *= 25000

    # inverse fourier transform each wire to get the noise
    noise_d = cp.real(cp.fft.ifft(noise_d, axis=0)).astype(cp.float32)
    return noise_d


@cuda.jit(fastmath=True)
def shuffle_noise_kern(noise: cp.ndarray, new_noise: cp.ndarray, indices: cp.ndarray, roll_amounts: cp.ndarray):
    """Shuffle the noise array.
    :param noise: The true (from gen_noise) noise array
    :param roll_amounts: For each wire, how much to roll that wire (ie, shift all noise on that wire up by
    `roll_amounts[w]` ticks and rollover resultant ticks back to the beginning of the wire)
    :param indices: The wire index in `new_noise` for each original wire in `noise`
    :param new_noise: Output noise array
    """
    w, t = cuda.grid(2)
    if w < nwires and t < nticks:
        # shuffle wires
        new_w = indices[w]
        # roll noise values along each wire
        new_t = (t + roll_amounts[w]) % nticks
        new_noise[new_w, new_t] = noise[w, t]


def shuffle_noise(rng: RandomState, noise: cp.ndarray):
    """Take a wire noise array from gen_noise and shuffle it to make a new noise array.

    Running 1664 ifft's is pretty slow, so we skip those by only generating "true" noise every few runs. Instead, this
    function will rearrange the order of the wires (since each has independent noise) and will roll the noise values
    along each wire by random amounts.

    :param rng: cupy rng state
    :param noise: A 2d cp.ndarray with shape (nwires, nticks) of noise generated by gen_noise
    :return: A new 2d cp.ndarray with shape (nwires, nticks)
    """
    new_noise = cp.empty_like(noise)
    # wire at noise[0] goes into new_noise[indices[0]]
    indices = cp.arange(0, nwires)
    rng.shuffle(indices)
    roll_amounts = rng.randint(0, nwires, size=nwires)
    threads = (16, 32)
    blocks = tuple(math.ceil(noise.shape[axis] / threads[axis]) for axis in [0, 1])
    shuffle_noise_kern[blocks, threads](noise, new_noise, indices, roll_amounts)
    return new_noise


@cp.fuse
def get_n_ionization_electrons(energy_loss: cp.float32) -> cp.int32:
    """GPU vectorized function to convert energy losses into #s of electrons that get detected
    :param energy_loss: energy loss in MeV
    :return: num of electrons that make it to readout
    """
    # convert to eV, get rid of negatives, and put in an upper threshold (otherwise, the long tail of the landau makes
    # some super huge values wash everything else out)
    # todo eventually ask Mastbaum if this cutoff makes sense
    energy_loss = cp.minimum(cp.maximum(1e6 * energy_loss, 0), energy_loss_cutoff)
    # number of elections ionized
    n_electrons = energy_loss / energy_per_electron
    # number of electrons that make it to readout
    return (n_electrons * survival_probability).astype(cp.int32)


MAX_IONIZATION_ELECTRONS = int(energy_loss_cutoff / energy_per_electron * survival_probability)


@cp.fuse
def get_sigma_x(x: cp.float32):
    """GPU vectorized function to convert distance to readout to Gaussian width of spread
    :param x: Distance from interaction to readout plane, cm
    :return: Gaussian width, cm
    """
    # drift time, μs
    t = x / v_d
    # equation 3.1 of https://iopscience.iop.org/article/10.1088/1748-0221/16/09/P09025/pdf
    # this is the width on the collection plane (μs)
    sigma_t = cp.sqrt(sigma_t0_2 + (2 * DL / v_d ** 2) * t)
    return sigma_t * v_d * 6  # 6 is a fudge factor


# see benchmark_n_chunks.png
CHUNKS = args.chunks
# closest power of 2 >= the sqrt of CHUNKS
adc_grids = 2 ** (math.ceil(math.sqrt(CHUNKS)) - 1).bit_length()
adc_blocks = math.ceil(CHUNKS / adc_grids)


@cuda.jit(fastmath=True)
def adcs_noise(y0: cp.float32, y_slope: cp.float32, ionization_electrons: cp.ndarray,
               sigma_xs: cp.ndarray, event_arrays: cp.ndarray, wire_random: cp.ndarray, tick_random: cp.ndarray):
    """
    :param y0: Start tick
    :param y_slope: Tick slope as z increases
    :param ionization_electrons: How many ionization electrons were generated at each simulation step along the detector
    :param sigma_xs: The width of the Gaussian at each sim step along the detector
    :param event_arrays: A 2d cp.ndarray, with shape `(CHUNKS, nwires, nticks)` to be filled. Calling this
    kernel essentially creates `CHUNKS` new event arrays, to be summed together afterwords
    :param tick_random: For each simulation step, an array of `MAX_IONIZATION_ELECTRONS` Gaussian wire offsets
    :param wire_random: For each simulation step, an array of `MAX_IONIZATION_ELECTRONS` Gaussian tick offsets
    """
    chunk = cuda.grid(1)
    if chunk < CHUNKS:
        for idx in range(chunk, n_steps, CHUNKS):
            z = idx * nwires / n_steps
            y = y0 + y_slope * z
            sigma_x = sigma_xs[idx]
            n_electrons = ionization_electrons[idx]
            # each ionization electron gets put to a random (wire, tick) offset
            for i in range(n_electrons):
                # todo I think both wire & tick need to be converted from cm to wire/tick
                # wire, gaussian distributed around the current wire (z)
                # wire = sigma_x * xoroshiro128p_normal_float32(rngs, idx)
                wire = sigma_x * wire_random[idx, i]
                wire += z
                wire = int(wire)

                # tick, gaussian distributed around the current tick (y) (this is kinda weird?)
                # tick = sigma_x * xoroshiro128p_normal_float32(rngs, idx)
                tick = sigma_x * tick_random[idx, i]
                # tick = tick / nticks * height
                tick += y
                # tick = tick / height * nticks
                tick = int(tick)

                # boolean logic is much faster than branching on GPU, and using atomic add was faster than normal add
                # add 1 to that index if it is in the bounds of the event
                cuda.atomic.add(event_arrays, (chunk, wire, tick), 0 <= wire < nwires and 0 <= tick < nticks)


def simulate_track(rng: RandomState, event: cp.ndarray, x0: int, y0: cp.float32, x1: int, y1: cp.float32,
                   event_arrays_d: cp.ndarray, time_all: bool = False):
    """Simulates a mCP crossing the detector, from (x0, y0) to (x1, y1)
    :param rng: cupy rng state
    :param event: A 2d (nwires, nticks) cp.ndarray of noise
    :param x0: Start wire
    :param y0: Start tick
    :param x1: End wire
    :param y1: End tick
    :param event_arrays_d: A 3d (CHUNKS, nwires, nticks) cp.ndarray, used as CHUNKS new event arrays that the event is
    written into before being combined into `event`
    :param time_all: whether to time the individual parts of this function
    """
    start = perf_counter_ns()
    # random energy losses
    # todo figure out better what sigma should be
    energy_losses_d = gpu_random.landau_rvs(n_steps, dE_mean, dE_mean / width_factor, rng)

    if time_all:
        cuda.synchronize()
    stop_energy_losses = perf_counter_ns()

    # get # of ionization electrons
    ionization_electrons_d = get_n_ionization_electrons(energy_losses_d)

    if time_all:
        cuda.synchronize()
    stop_ionization_n = perf_counter_ns()

    # get sigma of normal distribution
    x_slope = (x1 - x0) / n_steps
    # distance to readout plane, cm
    xs_d = x0 + cp.arange(n_steps, dtype=cp.float32) * x_slope
    sigma_xs_d = get_sigma_x(xs_d)

    if time_all:
        cuda.synchronize()
    stop_sigmas = perf_counter_ns()

    # get individual
    y_slope = (y1 - y0) / n_steps

    # reduce
    wire_random_d = rng.normal(size=(n_steps, MAX_IONIZATION_ELECTRONS), dtype=cp.float32)
    tick_random_d = rng.normal(size=(n_steps, MAX_IONIZATION_ELECTRONS), dtype=cp.float32)
    # reset all `CHUNKS` of the temporary arrays
    event_arrays_d.fill(0)
    # simulate the interactions
    adcs_noise[adc_grids, adc_blocks](y0, y_slope, ionization_electrons_d, sigma_xs_d, event_arrays_d, wire_random_d,
                                      tick_random_d)

    if time_all:
        cuda.synchronize()
    stop_adcs = perf_counter_ns()

    # combine all `CHUNKS` arrays into one
    reduced = event_arrays_d.sum(axis=0)

    if time_all:
        cuda.synchronize()
    stop_reduction = perf_counter_ns()

    # convert number of electrons into number of adc
    reduced /= electrons_per_adc
    # add this track to the event
    event += reduced

    if time_all:
        cuda.synchronize()
    stop = perf_counter_ns()
    timings = [stop_energy_losses - start,
               stop_ionization_n - stop_energy_losses,
               stop_sigmas - stop_ionization_n,
               stop_adcs - stop_sigmas,
               stop_reduction - stop_adcs,
               stop - stop_reduction]
    return timings


@cuda.jit(device=True)
def sum_along_line(x0: cp.float32, y0: cp.float32, x1: cp.float32, y1: cp.float32, event: cp.ndarray, aa: bool,
                   normalize: bool):
    """Sum value at each point in `event` connecting (x0, y0) to (x1, y1).
    Implementation from paper cited in `skimage.draw.line_aa`: 
    Listing 16 of http://members.chello.at/easyfilter/Bresenham.pdf

    :param x0: Start x (wire) coordinate
    :param y0: Start y (ticK) coordinate
    :param x1: End x (wire) coordinate
    :param y1: End y (tick) coordinate
    :param event: A 2d `(nwires, nticks)` cp.ndarray with at least one track
    :param aa: Whether to anti-alias line
    :param normalize: Whether to normalize sum by line length
    :return: 
    """
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


# see benchmark_n_threads.png
# 1024 is the max # of threads per block
threads = args.threads
# how many end tick values each block is able to calculate (170) (all these numbers are as if threads == 1024)
ticks_per_block = threads // tick_separation
# the last thread per block, ignores remaining `x` threads if `x<tick_separation` (1020)
max_thread_idx = ticks_per_block * tick_separation
# number of blocks (20)
blocks = math.ceil(nticks / ticks_per_block)
# target min height in ticks
y_min_t = y_min / dtick


@cuda.jit
def track_sums_kern(event: cp.ndarray, sums: cp.ndarray, aa: bool, normalize: bool):
    """Kernel to calculate sum along each line in event pointing back to the target.

    This kernel only calculates sums at pairs that can point back to the target, and doesn't touch the rest.

    :param event: A 2d (nwires, nticks) cp.ndarray with at least one track
    :param sums: A 2d (nticks, nticks) cp.ndarray to be filled in
    :param aa: Whether to antialias line
    :param normalize: Whether to normalize sums by line length
    """
    # tid runs from `0` to `1023` (`threads - 1`),
    # bid runs from `0` to `19` (`blocks - 1`)
    # cuda.grid(1) == cuda.threadIdx + cuda.blockIdx.x * cuda.blockDim.x
    tid = cuda.threadIdx.x
    # check that the thread in bounds
    if tid < max_thread_idx:
        # for these comments, threads==1024 and tick_separation==6
        # end tick is (0 to 169, 6 at a time) + (one of 0, 170, 340, 510, etc)
        # so (0) + (0) = 0 happens 6 times, then (1) + (0) = 1 happens 6 times, ..., (169) + (0) = 169 happens 6 times
        # then, block 1 happens, so each of those is increased by 170
        end = (tid // tick_separation) + (cuda.blockIdx.x * ticks_per_block)
        # y = b + mx, where x = end - y_min_t (?)
        start = y_min_t + (abs(target) / (abs(target) + length_wires)) * (end - y_min_t)
        # round start, and then get increase it by 0, 1, 2, 3, 4, or 5 to get from y_min_t to y_max_t
        start = round(start) + tid % tick_separation
        if start < nticks:
            sums[start, end] = sum_along_line(0, start, nwires - 1, end, event, aa, normalize)


def track_sums(event: cp.ndarray):
    """
    Get sums along the lines connecting each (start, end) pair that is on a line pointing back to the target. Pairs not
    pointing back to the target get a value of `-1`.

    :param event: A (nwires, nticks) cp.ndarray with at least one track.
    :return: A (nticks, nticks) cp.ndarray of sums along lines along (start, end)
    """
    sums_d = cp.full((nticks, nticks), -1, cp.float32)
    # always antialias line, always normalize the sums
    track_sums_kern[blocks, threads](event, sums_d, True, True)
    return sums_d


np_rng = np.random.default_rng()
pairs_seed = np_rng.integers(np.iinfo(np.uint64).max, dtype=np.uint64)

# method = args.rng
cp_rng = cp.random.RandomState()

# track_rng = create_xoroshiro128p_states(n_steps, seed=track_seed)
timing = np.empty((6, iters))
n_times_noise_used = 5

time_all = args.no_time_all and mode != 'run'

# preallocate some stuff
if mode == 'profile':
    print('START!')
event_arrays_d = cp.empty((CHUNKS, nwires, nticks), cp.float32)

# compile all kernels
y0, y1 = np.array([2021, 2109], dtype=cp.float32)
if mode == 'profile':
    print('START: gen_noise')
event = gen_noise(cp_rng)
if mode == 'profile':
    print('START: shuffle_noise')
shuffle_noise(cp_rng, event)
if mode == 'profile':
    print('START: simulate_track')
simulate_track(cp_rng, event, 0, y0, n_steps - 1, y1, event_arrays_d)
if mode == 'profile':
    print('START: track_sums')
sums = track_sums(event)

# plt.imshow(sums.get().T, interpolation='none', cmap='winter')
# # plt.colorbar()
# # plt.axis('off')
# plt.tight_layout()
# plt.savefig('sums.png', dpi=500)
# plt.show()

if mode == 'profile':
    print('START: peak_local_max')
peaks = peak_local_max(sums, min_distance=1, num_peaks=1)

if mode == 'profile':
    print("DONE!")
    exit(0)

if mode == 'bench' or mode == 'run':
    print(f'{"Timing" if mode == "bench" else "Running"} {"individual parts, " if time_all else ""}{iters} iterations')

sim_track_timings = np.empty((6, iters))

pairs_rng = create_xoroshiro128p_states(iters, seed=pairs_seed)
truth_pairs = get_start_end_pairs(iters, pairs_rng).get()
found_pairs = np.empty_like(truth_pairs)

stored_noise = None
for i in range(iters):
    if mode == 'run':
        if i % 100 == 0:
            # \r so the line just updates
            print(f'{i / iters * 100:.1f}% done   ', end='\r')
        elif i == iters - 1:
            print('100% done   ')

    start = perf_counter_ns()

    if i % n_times_noise_used == 0:
        event = gen_noise(cp_rng)
    else:
        event = shuffle_noise(cp_rng, stored_noise)
    if time_all:
        cuda.synchronize()
    stop_gen_noise = perf_counter_ns()

    y0, y1 = truth_pairs[i]
    sim_track_timing = simulate_track(cp_rng, event, 0, y0, n_steps - 1, y1, event_arrays_d, time_all)
    if time_all:
        for idx, t in enumerate(sim_track_timing):
            sim_track_timings[idx, i] = t
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

    found_peak = peaks[0].get()
    found_pairs[i] = found_peak
    if time_all:
        cuda.synchronize()
    stop_get = perf_counter_ns()

    stop = perf_counter_ns()
    if i % n_times_noise_used == 0:
        stored_noise = event

    # print(peaks)
    timing[0, i] = stop - start
    timing[1, i] = stop_gen_noise - start
    timing[2, i] = stop_sim_track - stop_gen_noise
    timing[3, i] = stop_track_sums - stop_sim_track
    timing[4, i] = stop_plm - stop_track_sums
    timing[5, i] = stop_get - stop_plm

timing *= 1e-6
sim_track_timings *= 1e-6

if mode == 'chunks':
    print(
        f"{CHUNKS} {sim_track_timings[3].mean():.3f} {sim_track_timings[3].std():.3f} {sim_track_timings[4].mean():.3f}"
        f" {sim_track_timings[4].std():.3f}")

if mode == 'threads':
    print(f"{threads} {blocks} {timing[3].mean():.3f} {timing[3].std():.3f}")

if mode == 'bench':
    print(f"              Elapsed time: {timing[0].mean():.3f} ± {timing[0].std():.3f} ms")
    if time_all:
        print(f"  Elapsed time (gen_noise): {timing[1].mean():.3f} ± {timing[1].std():.3f} ms")
        print(f"  Elapsed time (sim_track): {timing[2].mean():.3f} ± {timing[2].std():.3f} ms")
        # print(f"    sim_track::energy_losses: {sim_track_timings[0].mean():.3f} ± {sim_track_timings[0].std():.3f}")
        # print(f"     sim_track::n_ionization: {sim_track_timings[1].mean():.3f} ± {sim_track_timings[1].std():.3f}")
        # print(f"           sim_track::sigmas: {sim_track_timings[2].mean():.3f} ± {sim_track_timings[2].std():.3f}")
        # print(f"             sim_track::adcs: {sim_track_timings[3].mean():.3f} ± {sim_track_timings[3].std():.3f}")
        # print(f"      sim_track::full_reduce: {sim_track_timings[4].mean():.3f} ± {sim_track_timings[4].std():.3f}")
        # print(f"     sim_track::add_to_event: {sim_track_timings[5].mean():.3f} ± {sim_track_timings[5].std():.3f}")
        print(f"  Elapsed time (track_sum): {timing[3].mean():.3f} ± {timing[3].std():.3f} ms")
        print(f"        Elapsed time (plm): {timing[4].mean():.3f} ± {timing[4].std():.3f} ms")
        print(f"        Elapsed time (get): {timing[5].mean():.3f} ± {timing[5].std():.3f} ms")

if mode == 'bench' or mode == 'run':
    dists = np.linalg.norm(found_pairs - truth_pairs, axis=1)
    print(f'z = {z}')
    print(f'len(dists) = {len(dists)}')
    dists = dists[dists < 5]
    print(f'len(dists) = {len(dists)}')

    print(f'np.min(dists) = {np.min(dists)}')
    print(f'np.max(dists) = {np.max(dists)}')
    fig, ax = plt.subplots()
    mu = dists.mean()
    sigma = dists.std()
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\sigma=%.2f$' % (sigma,)))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.81, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    ax.hist(dists, edgecolor='white')
    ax.set_title(f'Distance from truth, {iters} points')
    ax.set_xlabel('Distance [cm]')
    ax.set_ylabel('Count')
    plt.show()
    plt.savefig(f"dist_q={z}.png")
