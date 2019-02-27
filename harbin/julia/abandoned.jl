

## create the hdf5 file with metadata
function createhdf5(year::Int, month::Int, h5path::String)
    isdir(h5path) || error("$h5path does not exist")
    filename = joinpath(h5path, @sprintf "trips-%d-%02d.h5" 2015 1)
    h5open(filename, "w") do file
        file["/meta/year"] = year
        file["/meta/month"] = month
    end
    filename
end


function trips2hdf5(ntrips::Int, trips::Vector{DataFrame}, h5file::String)
    """
    Save an array of dataframes (returned by `gpspoints2trips`) into a hdf5 file.

    trips2hdf5(ntrips::Int, trips::Vector{DataFrame}, h5file::String)
    Input:
    ntrips: The base of trip id
    trips: An array of dataframe
    h5file: hdf5 file name
    ---
    Return:
    The number of trips saved into h5file
    """
    isfile(h5file) || error("$h5file does not exist")
    n = length(trips)
    h5open(h5file, "r+") do f
        sort!(trips, by = trip -> trip[1,:gpstime])
        for i = 1:n
            f["/trip/$(ntrips+i)/lat"] = trips[i][:latitude]
            f["/trip/$(ntrips+i)/lon"] = trips[i][:longitude]
            f["/trip/$(ntrips+i)/tms"] = Dates.datetime2unix.(trips[i][:gpstime])
            f["/trip/$(ntrips+i)/speed"] = trips[i][:speed]
            f["/trip/$(ntrips+i)/devid"] = trips[i][:devid][1]
            #if i >= 10 break end
        end
    end
    n
end


function csvfiles2hdf5(fnames::Vector{String}, h5file::String)
    """
    Save a collection of csv files into one hdf5 file.

    Input:
    fnames: An array of csv file names
    h5file: hdf5 file name
    ---
    Return:
    The number of trips saved into h5file
    """
    ntrips = 0
    sort!(fnames, by = fname->basename(fname))
    for fname in fnames
        println("Processing $(basename(fname))...")
        df = read_cleandata(fname)
        trips = gpspoints2trips(df, 60, 10)
        n = trips2hdf5(ntrips, trips, h5file)
        println("Processed $(basename(fname)) with $n trips.")
        ntrips = ntrips + n
    end
    h5open(h5file, "r+") do f
        f["/meta/ntrips"] = ntrips
    end
    ntrips
end


function packmonth2hdf5(year::Int, month::Int, csvpath::String, h5path::String)
    """
    Pack all csv files in csvpath which fall into year-month into a hdf5 file which
    will be created in h5path (destroy the file if it has existed).

    Input:
    csvpath: The path of csv files
    h5path: The path of hdf5 file (created in the path)
    ---
    Return:
    The number of trips saved
    """
    isdir(csvpath) || error("$csvpath does not exist")
    isdir(h5path) || error("$h5path does not exist")

    fnames = filter(x -> parse(split(x, "_")[3][3:4]) == month, readdir(csvpath))
    fnames = map(x->joinpath(csvpath, x), fnames)
    h5file = createhdf5(year, month, h5path)
    csvfiles2hdf5(fnames, h5file)
end

function _bisearch_first{T<:Real}(a::Vector{T}, x::T, Δ::T)
    """
    Find the index of the first element a[i] s.t. a[i]-x >= Δ
    """
    li, ri = 0, length(a)+1
    while li < ri-1
        i = (li+ri) >>> 1
        ## if a[i] is too large then move searching range towards left
        if a[i] - x >= Δ
            ri = i
        else
            li = i
        end
    end
    ri
end

function _bisearch_last{T<:Real}(a::Vector{T}, x::T, Δ::T)
    """
    Find the index of the last element a[i] s.t. x - a[i] >= Δ
    """
    li, ri = 0, length(a)+1
    while li < ri-1
        i = (li+ri) >>> 1
        ## if a[i] is too small then move searching range towards right
        if x - a[i] >= Δ
            li = i
        else
            ri = i
        end
    end
    li
end

## Return the index of the first element of a such that a[i] >= x+Δ
bisearch_first{T<:Real}(a::Vector{T}, x::T, Δ::T) = searchsortedfirst(a, x+Δ)
## Return the index of the last element of a such that a[i] <= x-Δ
bisearch_last{T<:Real}(a::Vector{T}, x::T, Δ::T) = searchsortedlast(a, x-Δ)

function createflowtensor!(region::SpatialRegion,
                           trips::Vector{Trip})

    """
    Create the tensor that counts the number of taxi inflowing and outflowing each
    cell in `region` using the `trips`.

    region.traffic[:, :, 1] counts the inflow
    region.traffic[:, :, 2] counts the outflow
    """
    region.traffic[:, :, 1:2] .= 0
    for trip in trips
        offsets, _ = rle(gps2regionOffset.(region, trip.lon, trip.lat))
        n = length(offsets)
        for i = 1:n
            x, y = offsets[i] .+ 1
            if i == 1
                region.traffic[y, x, 2] += 1
            elseif i == n
                region.traffic[y, x, 1] += 1
            else
                region.traffic[y, x, 1] += 1
                region.traffic[y, x, 2] += 1
            end
        end
    end
end

function linear_interpolate(p1::Tuple{T,T,T}, p2::Tuple{T,T,T}, Δ) where T
    sx, sy, st = p1
    ex, ey, et = p2
    d = euclidean([sx, sy], [ex, ey])
    ## inserted points
    p = mod(d, Δ) == 0 ? collect(0:Δ:d)/d : push!(collect(0:Δ:d), d)/d
    ## direction
    vx, vy = ex - sx, ey - sy
    v̄ = d / (et - st) # average speed
    collect(zip(sx + vx * p, sy + vy * p, fill(v̄, length(p))))
end

function linear_interpolate(trip::Vector{Tuple{T,T,T}}, Δ=200.) where T
    """
    Interpolate a trip linearly.

    Input:
      trip: a vector of Webmercator coordinates with timestampes.
    Return:
      fine_trip: a vector of Webmercator coordinates with mean speed.
    """
    fine_trip = Tuple{T,T,T}[]
    for i = 2:length(trip)
        points = linear_interpolate(trip[i-1], trip[i], Δ)
        i == 2 ? append!(fine_trip, points) : append!(fine_trip, points[2:end])
    end
    fine_trip
end

function linear_interpolate(trip::Vector{Tuple{T,T}}, tms::Vector{T}, Δ=200.) where T
    # x, y = collect(zip(trip...))
    x, y = first.(trip), last.(trip)
    fine_trip = linear_interpolate(collect(zip(x, y, tms)), Δ)
    (x -> x[1:2]).(fine_trip), last.(fine_trip)
end








function logsoftmax!(out::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxForward(xs, out, algorithm=CUDNN_SOFTMAX_LOG)
  return out
end

function ∇logsoftmax!(out::CuVecOrMat{T}, Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat
  cudnnSoftmaxBackward(logsoftmax(xs), Δ, out, algorithm=CUDNN_SOFTMAX_LOG)
  return out
end

∇logsoftmax(Δ::CuVecOrMat{T}, xs::CuVecOrMat{T}) where T<:CUDNNFloat = ∇logsoftmax!(similar(xs), Δ, xs)


"""
    logσ(x)

Return `log(σ(x))` which is computed in a numerically stable way.

    julia> logσ(0.)
    -0.6931471805599453
    julia> logσ.([-100, -10, 100.])
    3-element Array{Float64,1}:
     -100.0
      -10.0
       -0.0
"""
function logσ(x)
    max_v = max(zero(x), -x)
    z = exp(-max_v) + exp(-x-max_v)
    -(max_v + log(z))
end

∇logσ(Δ, x) = Δ * (1 - σ(x))

const logsigmoid = logσ
const ∇logsigmoid = ∇logσ

@test logsigmoid.(xs) ≈ log.(sigmoid.(xs))

for T in [:Float32, :Float64]
    @eval @test logsigmoid.($T[-100_000, 100_000.]) ≈ $T[-100_000, 0.]
end

"""
    logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)

`logitcrossentropy` takes the logits of the output probability distribution `logŷ` and
the target probability distribution `y` as the inputs to compute the cross entropy loss.
It is mathematically equivalent to the combination of `softmax(logŷ)` and `crossentropy`,
i.e., `crossentropy(softmax(logŷ), y)`, but it is more numerically stable than the former.

    julia> srand(123);
    julia> x = randn(5, 4);
    julia> y = rand(10, 4);
    julia> y = y ./ sum(y, 1);
    julia> m = Dense(5, 10);
    julia> logŷ = m(x);
    julia> Flux.logitcrossentropy(logŷ, y)
    Tracked 0-dimensional Array{Float64,0}:
    2.44887
"""
function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
  return -sum(y .* logsoftmax(logŷ) .* weight) / size(y, 2)
end


"""
    binarycrossentropy(ŷ, y)

Return `-y*log(ŷ) - (1-y)*log(1-ŷ)`.

    julia> binarycrossentropy.(σ.([-1.1491, 0.8619, 0.3127]), [1, 1, 0.])
    3-element Array{Float64,1}:
    1.4244
    0.352317
    0.86167
"""
binarycrossentropy(ŷ, y) = -y*log_fast(ŷ) - (1 - y)*log_fast(1 - ŷ)

"""
    logitbinarycrossentropy(logŷ, y)

`logitbinarycrossentropy(logŷ, y)` is mathematically equivalent to `binarycrossentropy(σ(logŷ), y)`
but it is more numerically stable.

    julia> logitbinarycrossentropy.([-1.1491, 0.8619, 0.3127], [1, 1, 0.])
    3-element Array{Float64,1}:
     1.4244
     0.352317
     0.86167
"""
logitbinarycrossentropy(logŷ, y) = (1 - y)*logŷ - logσ(logŷ)

logŷ, y = randn(3), rand(3)
logitbinarycrossentropy.(logŷ, y) ≈ binarycrossentropy.(σ.(logŷ), y)
binarycrossentropy.(σ.(logŷ), y) ≈ -y.*log.(σ.(logŷ)) - (1 - y).*log.(1 - σ.(logŷ))
