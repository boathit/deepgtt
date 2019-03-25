
# DeepGTT

This repository holds the code used in our **WWW-19** paper: [Learning Travel Time Distributions with Deep Generative Model](http://www.ntu.edu.sg/home/lixiucheng/pdfs/www19-deepgtt.pdf).

## Requirements

* Ubuntu OS (16.04 and 18.04 are tested)
* [Julia](https://julialang.org/downloads/) >= 1.0
* Python >= 3.6
* PyTorch >= 0.4 (both 0.4 and 1.0 are tested)

Please refer to the source code to install the required packages in both Julia and Python. You can install packages for Julia in shell as

```bash
julia -e 'using Pkg; Pkg.add("HDF5"); Pkg.add("CSV"); Pkg.add("DataFrames"); Pkg.add("Distances"); Pkg.add("StatsBase"); Pkg.add("JSON"); Pkg.add("Lazy"); Pkg.add("JLD2"); Pkg.add("ArgParse")'
```

## Dataset

The dataset contains 1 million+ trips collected by 1,3000+ taxi cabs during 5 days. This dataset is a subset of the one we used in the paper, but it suffices to reproduce the results that are very close to what we have reported in the paper.

```bash
git clone https://github.com/boathit/deepgtt

cd deepgtt && mkdir -p data/h5path data/jldpath data/trainpath data/validpath data/testpath
```

Download the [dataset](https://drive.google.com/open?id=1tdgarnn28CM01o9hbeKLUiJ1o1lskrqA) and put the extracted `*.h5` files into `deepgtt/data/h5path`.

### Data format

Each h5 file contains `n` trips of the day. For each trip, it has three fields `lon` (longitude), `lat` (latitude), `tms` (timestamp). You can read the h5 file using the [`readtripsh5`](https://github.com/boathit/deepgtt/blob/master/harbin/julia/Trip.jl#L28) function in Julia. If you want to use your own data, you can also refer to [`readtripsh5`](https://github.com/boathit/deepgtt/blob/master/harbin/julia/Trip.jl#L28) to dump your trajectories into the required hdf5 files.

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

To run the python code, make sure you have set up the road network postgresql server by referring to the map server setup in [barefoot](https://github.com/boathit/barefoot). The road network server (see this [file](https://github.com/boathit/deepgtt/blob/master/harbin/python/db_utils.py#L8)) is used to provide road segment features for the model.

```bash
cd deepgtt/harbin/python

python train.py -trainpath ../data/trainpath -validpath ../data/validpath -kl_decay 0.0 -use_selu -random_emit
```

## Testing

```bash
python estimate.py -testpath ../data/testpath
```

## Reference

```
@inproceedings{www19xc,
  author    = {Xiucheng Li and
               Gao Cong and
               Aixin Sun and
               Yun Cheng},
  title     = {Learning Travel Time Distributions with Deep Generative Model},
  booktitle = {Proceedings of the 2019 World Wide Web Conference on World Wide Web,
               {WWW} 2019, San Francisco, California, May 13-17, 2019},
  year      = {2019},
}
```
