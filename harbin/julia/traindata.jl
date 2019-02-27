using HDF5, JSON, JLD2, FileIO, Dates
include("traffic.jl")
include("Trip.jl")
include("SpatialRegion.jl")

function collectslotdata(region::SpatialRegion, trips::Vector{Trip}, tms::T) where T
    """
    Collect the trips in the past 30 minutes of `tms` to create the traffic tensor
    along with the trips in the future 40 minutes.
    """
    slotsubtrips = timeslotsubtrips(trips, tms-30*60, tms)
    createflowtensor!(region, slotsubtrips)
    slottrips = timeslottrips(trips, tms, tms+46*60)
    ##
    slottrips = filter(t->t.tms[end]-t.tms[1]>=7*60, slottrips)
    copy(region.S), slottrips
end

function savetraindata(h5file::String, region::SpatialRegion, trips::Vector{Trip}, stms::T) where T
    """
    trips: all trips in the day
    stms: start unix time of the day
    """
    hasratio = isdefined(trips[1], :endpointsratio)
    h5open(h5file, "w") do f
        range = 30*60:20*60:24*3600 # maybe change 20 mins to 10 mins
        for (slot, tms) in enumerate(range)
            S, slottrips = collectslotdata(region, trips, stms+tms)
            f["/$slot/S"] = S
            f["/$slot/ntrips"] = length(slottrips)
            for i = 1:length(slottrips)
                f["/$slot/trip/$i"] = slottrips[i].roads
                f["/$slot/time/$i"] = (slottrips[i].tms[end]-slottrips[i].tms[1])/60.0
                ## origin and destination
                f["/$slot/orig/$i"] = [slottrips[i].lon[1]-region.minlon, slottrips[i].lat[1]-region.minlat]
                f["/$slot/dest/$i"] = [slottrips[i].lon[end]-region.minlon, slottrips[i].lat[end]-region.minlat]
                f["/$slot/ratio/$i"] = hasratio ? slottrips[i].endpointsratio : [1.0, 1.0]
                ## path distance
                f["/$slot/distance/$i"] = pathdistance(slottrips[i])
            end
        end
    end
end

function savetraindata(h5path::String, region::SpatialRegion, jldfile::String)
    """
    Dump the trips in `jldfile` into train data.
    """
    ymd = basename(jldfile) |> splitext |> first |> x->split(x, "_") |> last
    m, d = parse(Int, ymd[3:4]), parse(Int, ymd[5:6])
    h5file = ymd * ".h5"
    ## Filtering out the trajectories with serious GPS errors
    trips = filter(trip -> length(trip.roads)>=5, load(jldfile, "trips"))
    stms = Dates.datetime2unix(DateTime(2015, m, d, 0, 0))
    savetraindata(joinpath(h5path, h5file), region, trips, stms)
end

function savetraindata(h5path::String, jldpath::String)
    """
    Dump all jldfiles into h5files (training data).
    """
    param  = JSON.parsefile("../hyper-parameters.json")
    region = param["region"]
    harbin = SpatialRegion{Float64}("harbin",
                                    region["minlon"], region["minlat"],
                                    region["maxlon"], region["maxlat"],
                                    region["cellsize"], region["cellsize"])
    fnames = readdir(jldpath)
    for fname in fnames
        println("saving $fname...")
        jldfile = joinpath(jldpath, fname)
        savetraindata(h5path, harbin, jldfile)
    end
end
