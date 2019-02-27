
# DeepGTT

This repository holds the code used in our **WWW-19** paper: [Learning Travel Time Distributions with Deep Generative Model]().

## Requirements

* [Julia](https://julialang.org/downloads/) >= 1.0
* Python >= 3.6
* PyTorch >= 0.4

Please refer to the source code to install the required packages in both Julia and Python.

## Dataset

The dataset contains 1 million+ trips collected by 1,3000+ taxi cabs during 5 days. This dataset is a subset of the one we used in the paper, but it suffices to reproduce the results that are very close to what we have reported in the paper.

Download the dataset and put the extracted `*.h5` files into `deepgtt/data/h5path`.

Each h5 file contains `n` trips of the day. For each trip, it has three fields `lon` (longitude), `lat` (latitude), `tms` (timestamp). You can read the h5 file using the `readtripsh5` function in Julia,

```julia
function readtripsh5(tripfile::String)
    """
    Read trips from hdf5file
    """
    trips = Trip[]
    h5open(tripfile, "r") do f
        ntrips = read(f["/meta/ntrips"])
        for i = 1:ntrips
            lon = read(f["/trip/$i/lon"])
            lat = read(f["/trip/$i/lat"])
            tms = read(f["/trip/$i/tms"])
            trip = Trip(lon, lat, tms)
            push!(trips, trip)
        end
    end
    trips
end
```

## Preprocessing

### Map matching

First, setting up the map server and matching server by referring to [barefoot](https://github.com/boathit/barefoot).

Then, matching the trips

```bash
cd deepgtt/harbin/julia

julia -p 6 mapmatch.jl --inputpath ../data/h5path --outputpath ../data/jldpath
```

where `6` is the number of cpu cores available in your machine.


### Generate training, validation and test data

```bash
julia gentraindata.jl --inputpath ../data/jldpath --outputpath ../data/trainpath

cd .. && mv data/trainpath/150106.h5 data/validpath && mv data/trainpath/150107.h5 data/testpath
```

## Training

```bash
cd deepgtt/harbin/python

python train.py -trainpath ../data/trainpath -validpath ../data/validpath -kl_decay 0.0 -use_selu -random_emit
```

## Testing

```bash
python estimate.py -testpath ../data/testpath
```
