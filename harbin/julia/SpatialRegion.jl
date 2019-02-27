
using Lazy:@>>
#include("util.jl")

"""
harbin = SpatialRegion{Float64}("harbin",
                                126.506130, 45.657920,
                                126.771862, 45.830905,
                                200., 200.)
"""
struct SpatialRegion{T<:AbstractFloat}
    name::String
    minlon::T
    minlat::T
    maxlon::T
    maxlat::T
    xstep::T
    ystep::T
    minx::T
    miny::T
    maxx::T
    maxy::T
    numx::Int
    numy::Int
    I # inflow
    O # outflow
    S # speed
    C # speed count

    function SpatialRegion{T}(name::String,
                              minlon::T,
                              minlat::T,
                              maxlon::T,
                              maxlat::T,
                              xstep::T,
                              ystep::T;
                              dim=3) where T
        minx, miny = gps2webmercator(minlon, minlat)
        maxx, maxy = gps2webmercator(maxlon, maxlat)
        numx = @>> round(maxx - minx, digits=6)/xstep ceil(Int)
        numy = @>> round(maxy - miny, digits=6)/ystep ceil(Int)
        new(name,
            minlon, minlat,
            maxlon, maxlat,
            xstep, ystep,
            minx, miny,
            maxx, maxy,
            numx, numy,
            zeros(Float32, numy, numx),
            zeros(Float32, numy, numx),
            zeros(Float32, numy, numx),
            zeros(Float32, numy, numx))
    end
end

#function createtraffic(region::SpatialRegion; dim=2)
#    region.traffic = zeros(Float32, numy, numx, dim)
#end

## Internal function
function coord2regionOffset(region::SpatialRegion, x::T, y::T) where T<:AbstractFloat
    """
    Given a region (city), converting the Web Mercator coordinate to offset tuple

    Input:
      region (SpatialRegion): `region` defines the range of the city
      x (Float)
      y (Float)
    Return:
      xoffset (Int): 0 <= xoffset < region.numx
      yoffset (Int): 0 <= yoffset < region.numy
    """
    xoffset = @>> round(x - region.minx, digits=6) / region.xstep floor(Int)
    yoffset = @>> round(y - region.miny, digits=6) / region.ystep floor(Int)
    xoffset, yoffset
    #yoffset * region.numx + xoffset
end

coord2regionOffset(region, xy) = coord2regionOffset(region, xy...)

## External function
function gps2regionOffset(region::SpatialRegion, lon::T, lat::T) where T<:AbstractFloat
    """
    Given a region (city), converting the GPS coordinate to the offset tuple
    """
    function isInRegion(region, lon, lat)
        region.minlon <= lon < region.maxlon &&
        region.minlat <= lat < region.maxlat
    end
    @assert isInRegion(region, lon, lat) "lon:$lon lat:$lat out of region:$(region.name)"
    @>> gps2webmercator(lon, lat) coord2regionOffset(region)
end

gps2regionOffset(region, gps) = gps2regionOffset(region, gps...)

function reset!(region::SpatialRegion)
    for field in [:I, :O, :S, :C]
        @eval $region.$field .= 0
    end
end
