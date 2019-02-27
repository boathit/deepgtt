using HDF5, CSV, DataFrames, Dates, Sockets
using Distances: euclidean
using StatsBase:rle
import JSON
include("util.jl")

mutable struct Trip{T<:AbstractFloat}
    lon::Vector{T}
    lat::Vector{T}
    tms::Vector{T}
    devid
    roads
end

Trip(lon, lat, tms) = Trip(lon, lat, tms, 0, nothing)
Trip(lon, lat, tms, devid) = Trip(lon, lat, tms, devid, nothing)

function Base.show(io::IO, t::Trip)
    print(io, "Trip: $(length(t.lon)) points")
end

Base.length(t::Trip) = length(t.lon)

Base.reverse(trip::Trip) = Trip(reverse(trip.lon), reverse(trip.lat), trip.tms)

#trip = Trip([1, 2, 3.], [1, 3, 2.], [3, 4, 5.])

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
            #i >= 10000 && break
        end
    end
    trips
end

function readtripscsv(tripfile::String, header=nothing)
    """
    Read trips from csvfile. If `header` is unspecified, the csvfile must contain
    columns [:devid, :tripid, :tms, :lon, :lat].
    """
    df = CSV.File(tripfile, header=header) |> DataFrame
    hasdevid = :devid in names(df)
    trips = Trip[]
    for sdf in groupby(df, :tripid)
        #sdf = df[df.tripid .== tripid[i], :]
        sdf = DataFrame(sdf)
        sort!(sdf, :tms)
        lon = convert(Array{Float64}, sdf.lon)
        lat = convert(Array{Float64}, sdf.lat)
        tms = convert(Array{Float64}, sdf.tms)
        devid = hasdevid ? first(sdf.devid) : nothing
        trip = Trip(lon, lat, tms, devid)
        push!(trips, trip)
    end
    trips
end

function pathdistance(trip::Trip)
    """
    The trip distance (km) in the road network
    """
    s = 0.0
    for i = 2:length(trip)
        px, py = gps2webmercator(trip.lon[i-1], trip.lat[i-1])
        cx, cy = gps2webmercator(trip.lon[i], trip.lat[i])
        s += euclidean([px, py], [cx, cy])
    end
    s / 1000.0
end


function isvalidtrip(trip::Trip)
    """
    Return true if the trip is a valid trip which maximum speed cannnot exceed 35
    otherwise return false.
    """
    for i = 2:length(trip.tms)
        px, py = gps2webmercator(trip.lon[i-1], trip.lat[i-1])
        cx, cy = gps2webmercator(trip.lon[i], trip.lat[i])
        euclidean([px, py], [cx, cy]) / (trip.tms[i] - trip.tms[i-1]) > 35 && return false
    end
    true
end

function timeslotsubtrips(trips::Vector{Trip}, stms::T, etms::T) where T
    """
    Return the sub-trips falling into time slot [stms, etms].
    """
    subtrips = Trip[]
    for trip in trips
        a, b = searchrange(trip.tms, stms, etms)
        if a < b
            subtrip = Trip(trip.lon[a:b], trip.lat[a:b], trip.tms[a:b], trip.devid)
            push!(subtrips, subtrip)
        end
    end
    subtrips
end

function timeslottrips(trips::Vector{Trip}, stms::T, etms::T, Δmins=5) where T
    """
    Return the trips whose start time falling into time slot [stms, etms].
    """
    filter(trips) do trip
        stms < trip.tms[1] < etms && etms + Δmins*60 > trip.tms[end]
    end
end

function trip2finetrip(trip::Trip, Δ=200)
    """
    Interpolating a coarse trip into a fine-grained trip
    Input:
      trip (Trip)
      Δ (Real): The minimum Euclidean distance between two consecutive points after interpolation.
    Output:
      A trip
    """
    finepoints, tms, _ = linear_interpolate(gps2webmercator.(trip.lon, trip.lat), trip.tms, Δ)
    x, y = map(first, finepoints), map(last, finepoints)
    gpspoints = webmercator2gps.(x, y)
    lon, lat = map(first, gpspoints), map(last, gpspoints)
    Trip(lon, lat, tms, trip.devid)
end

function removeredundantpoints(trip::Trip, δ=10)
    """
    Remove the redundant sampling points in a trip. A point is redundant if the timestamp gap
    of it and its last point is smaller than δ. The function is applied to a trip before submitting
    it to the map matcher in order to avoid unnecessary computation.
    Input:
      trip (Trip)
      δ (Real): the minimum gap whose the unit is second
    Output:
      _ (Trip)
    """
    ind = Int[1]
    for i = 2:length(trip)
        if trip.tms[i]-trip.tms[ind[end]] >= δ
            push!(ind, i)
        end
    end
    Trip(trip.lon[ind], trip.lat[ind], trip.tms[ind], trip.devid)
end

###############################################################################
function trip2geojson(trip::Trip)
    points = [Dict("type"=>"Feature",
                   "geometry"=>Dict("type"=>"Point",
                                    "coordinates"=>[trip.lon[i], trip.lat[i]]))
                   for i = 1:length(trip)]
    Dict("type"=>"FeatureCollection",
         "features"=>points)
end

function trip2json(trip::Trip)
    """
    Mapping a trip to the json format required by the map matcher
    Input:
      trip (Trip)
    Output:
      _ (Dict)
    """
    js = Dict{String, String}[]
    for i = 1:length(trip)
        lon, lat = trip.lon[i], trip.lat[i]
        tms = Dates.format(Dates.unix2datetime(trip.tms[i]), "yyyy-mm-dd HH:MM:SS")
        push!(js, Dict("point"=>"POINT($lon $lat)",
                       "time"=> "$(tms)+0800",
                       "id"=>"\\x0001"))
    end
    js
end

function matchtrip(trip::Trip)
    """
    Matching a trip by submitting it to the match server.

    Input:
      trip (Trip)
    Output:
      result (Vector): an array of dict
    """
    #js = trip2finetrip(trip, 200) |> removeredundantpoints |> trip2json
    js = trip |> removeredundantpoints |> trip2json
    request = Dict("format"=>"slimjson", "request"=>js) |> JSON.json
    clientside = connect("localhost", 1234)
    try
        message = request * "\n"
        write(clientside, message)
        response = readline(clientside)
        response == "SUCCESS" ? readline(clientside) |> JSON.parse : []
    finally
        close(clientside)
    end
end

function trip2roads(trip::Trip)
    """
    Mapping a trip to a sequence of road segments.
    """
    result = matchtrip(trip)
    roads, _ = map(d->get(d, "road", -1), result) |> rle
    roads
end
