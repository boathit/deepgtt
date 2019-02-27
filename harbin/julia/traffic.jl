
include("SpatialRegion.jl")
include("Trip.jl")
include("util.jl")

function createflowtensor!(region::SpatialRegion,
                           trips::Vector{Trip})

    """
    Create the tensor that counts the number of taxi inflowing and outflowing each
    cell in `region` using the `trips`.

    region.I counts the inflow
    region.O counts the outflow
    region.S stores the mean speed
    region.C counts the speed
    """
    function normalizex!(X)
        X ./= sum(X)
        X ./= maximum(X)
    end

    reset!(region)
    for trip in trips
        fine_trip, _, v̄ = linear_interpolate(gps2webmercator.(trip.lon, trip.lat), trip.tms)
        for i = 2:length(fine_trip)
            px, py = coord2regionOffset(region, fine_trip[i-1][1:2]...) .+ 1
            cx, cy = coord2regionOffset(region, fine_trip[i][1:2]...) .+ 1
            if cx ≠ px || cy ≠ py
                region.O[py, px] += 1 # outflow
                region.I[cy, cx] += 1 # inflow
                region.S[cy, cx] += v̄[i] # speed
                region.C[cy, cx] += 1
            elseif v̄[i] ≠ v̄[i-1] # mean speed changes so we count it
                region.S[cy, cx] += v̄[i] # speed
                region.C[cy, cx] += 1
            end
        end
    end
    normalizex!(region.I)
    normalizex!(region.O)
    idx = region.C .> 0
    region.S[idx] ./= region.C[idx]
end

# harbin = SpatialRegion{Float64}("harbin",
#                                 126.506130, 45.657920,
#                                 126.771862, 45.830905,
#                                 200., 200.)
# reset!(harbin)
