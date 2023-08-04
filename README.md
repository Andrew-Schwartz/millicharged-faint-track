# Millicharged Particle Faint Track

Simulate tracks from a millicharged particle (mCP) crossing the SBND detector, and detect those tracks.

See the [technote](technote.md) for a high level description of the design/algorithms/optimization strategies.

## Run configurations

There are two main files. The main file is [faint_tracks.py](faint_tracks.py), which has all the kernel definitions
and different run configurations for benchmarking, profiling, etc. The [notebook](faint_tracks.ipynb) is where I tested
parts individually, and not up to date.

### faint_tracks.py

Required command line arguments:

* `chunks` - How many chunks to use for `simulate_tracks`. It is useful to benchmark different values as it has a large
  influence on the runtime of `simulate_tracks`. See [this plot](images/details/benchmark_n_chunks.png) for
  the effect of different sizes for my configuration.
* `threads` - How many threads to use for `track_sums`. Can be 1024 at max. It is useful to benchmark different values
  as it has a large
  influence on the runtime of `track_sums`. See [this plot](images/details/benchmark_n_threads.png) for the effect of
  different #'s of threads.

For example, `python faint_tracks.py 128 32`.

Optional command line arguments:

* `--mode` - What mode to run. The options are
    * `bench` (default) - Run a bunch of iterations of the algorithm, print timing information, and save a histogram of
      the distances between found and true start/end pairs.
    * `profile` - Use this when profiling with `ncu`, to just run each gpu kernel once. Otherwise, it'll take forever to
      profile since each gpu function gets way slower.
    * `chunks` - Use this for testing different values of `chunks`, will just
      print `chunks mean1 stddev1 mean2 stddev2` (for `simulate_tracks`, 1 is calculating arrays, 2 is combining them).
    * `threads` -Use this for testing different values of `threads`, will just print `threads blocks mean stddev` (
      for `track_sums`).
    * `run` - Like bench, but for large numbers of events. Will print 
* `--no-time-all` - For `bench` mode, use this to just time the total elapsed time and not each individual part.
* `--iters` - How many iterations to run (default 100) for `bench`, `chunks`, or `threads`.
* `-z` - The charge of the particle to simulate (defaults to 1.0).
* `--gpu` - Select which gpu to use, zero indexed. Use `nvidia-smi` to see details about available gpus. Defaults to `3`
  because currently on `spsfarm`, the other 3 gpus are stuck in performance mode P2 (second highest), meaning they
  aren't able to cool down when not in use, so they are slower.