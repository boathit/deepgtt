## Transform the gps points in the csv files into trips in hdf5 files
## The parameters of minimun trip length and minimum time gap are set in
## function csvfile2hdf5

using DataFrames, CSV, HDF5

@everywhere include("parameters.jl")
@everywhere include("h5util.jl")

@everywhere function csvfile2hdf5(fname::String, h5path::String; verbose=false)
    """
    Transform the raw gps points in a csv file into trips
    Each trip contains:
      devid (Int)
      lat (Vector{Float})
      lon (Vector{Float})
      tms (Vector{Float}): timestamp
      speed (Vector{Float})


    Input:
      fname (String): csv file name
      h5path (String): The path of hdf5 file (the name of hdf5 file is determined
        by the csv file name)
    Output:
      n (Int): The number of trips that obtained
    """
    verbose && println("Processing $(basename(fname))...")
    h5file = split(basename(fname), ['_', '.'])[3]
    h5file = joinpath(h5path, "trips_$h5file.h5")
    df = read_cleandata(fname)
    trips = gpspoints2trips(df, MAX_GAP_SECONDS, MIN_DURATION_MINUTES)
    sort!(trips, by = trip -> trip[1,:gpstime])
    n = length(trips)
    h5open(h5file, "w") do f
        for i = 1:n
            tms = Dates.datetime2unix.(trips[i][:gpstime])
            f["/trip/$i/lat"] = trips[i][:latitude]
            f["/trip/$i/lon"] = trips[i][:longitude]
            f["/trip/$i/tms"] = tms
            f["/trip/$i/speed"] = trips[i][:speed]
            f["/trip/$i/devid"] = trips[i][:devid][1]
            #f["/trip/$i/mintms"] = tms[1]
            #f["/trip/$i/maxtms"] = tms[2]
            #if i >= 10 break end
        end
        f["/meta/ntrips"] = n
    end
    verbose && println("Processed $(basename(fname)) with $n trips.")
    n
end

function pcsvfiles2hdf5s(fnames::Vector{String}, h5path::String)
    """
    Parallel processing the csv files.

    Input:
      fnames (Vector{String}): csv files
      h5path (String): The path of hdf5
    """
    # @parallel (+) for fname in fnames
    #     csvfile2hdf5(fname, h5path, verbose=true)
    # end
    pmap(fname->csvfile2hdf5(fname, h5path, verbose=true), fnames)
end

function pmonth(month::Int, csvpath::String, h5path::String)
    """
    Transforming all month gps points data in `csvpath` into trips.
    """
    isdir(csvpath) || error("$csvpath does not exist")
    isdir(h5path) || error("$h5path does not exist")

    fnames = filter(x -> parse(split(x, "_")[3][3:4]) == month, readdir(csvpath))
    fnames = map(x->joinpath(csvpath, x), fnames)
    pcsvfiles2hdf5s(fnames, h5path)
end

csvpath = "/home/xiucheng/data-backup/bigtable/2015-taxi/data/cleandata"
h5path = joinpath(dirname(csvpath), "h5path")

#fnames = ["GPSHIS_DAY_150101.csv", "GPSHIS_DAY_150102.csv", "GPSHIS_DAY_150103.csv", "GPSHIS_DAY_150104.csv"]
#fnames = map(x->joinpath(csvpath, x), fnames)
#for fname in fnames
#    @time csvfile2hdf5(fname, "/tmp")
#end
#@time pcsvfiles2hdf5s(fnames, "/tmp/")

@time pmonth(1, csvpath, h5path)
