using Distances: euclidean
using DataFrames: DataFrame

function gps2webmercator(lon::T, lat::T) where T<:AbstractFloat
    """
    Converting GPS coordinate to Web Mercator coordinate
    """
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = sin(north)
    semimajoraxis * east, 3189068.5 * log((1 + t) / (1 - t))
end

function webmercator2gps(x::T, y::T) where T<:AbstractFloat
    """
    Converting Web Mercator coordinate to GPS coordinate
    """
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = exp(y / 3189068.5)
    lat = asin((t - 1) / (t + 1)) / 0.017453292519943295
    lon, lat
end

gps2webmercator(p) = gps2webmercator(p...)
webmercator2gps(p) = webmercator2gps(p...)

function searchrange(a::Vector{T}, sr::T, er::T) where T<:Real
    """
    Return the start and end index of element in `a` that falls into range [sr, er],
    and `a` is assumed to be sorted in ascending order.
    """
    searchsortedfirst(a, sr), searchsortedlast(a, er)
end

function linear_interpolate(p1::Tuple{T,T}, p2::Tuple{T,T}, t1, t2, Δ) where T
    """
    Insert n points between p1 and p2 where n = d(p1, p2) / Δ, return the interpolated
    n+2 points and the mean speed.
    """
    ϵ = 1e-8
    sx, sy = p1
    ex, ey = p2

    d = euclidean([sx, sy], [ex, ey])
    ## inserted points
    p = mod(d, Δ) == 0 ? collect(0:Δ:d)/(d+ϵ) : push!(collect(0:Δ:d), d)/(d+ϵ)
    ## direction
    vx, vy = ex - sx, ey - sy
    v̄ = d / (t2 - t1) # average speed
    collect(zip(sx .+ vx * p, sy .+ vy * p)), t1 .+ (t2-t1) * p, fill(v̄, length(p))
end


function linear_interpolate(trip::Vector{Tuple{T,T}}, tms::Vector, Δ=200.) where T
    """
    Interpolate a trip linearly.

    Input:
      trip: a vector of Webmercator coordinates with timestampes.
    Return:
      fine_trip: a vector of Webmercator coordinates with mean speed.
    """
    fine_trip, ts, vs = Tuple{T,T}[], T[], T[]
    for i = 2:length(trip)
        points, t, v = linear_interpolate(trip[i-1], trip[i], tms[i-1], tms[i], Δ)
        i == 2 ? (append!(fine_trip, points); append!(ts, t); append!(vs, v)) :
                 (append!(fine_trip, points[2:end]); append!(ts, t[2:end]); append!(vs, v[2:end]))
    end
    fine_trip, ts, vs
end


function matched2geojson(dict::Dict)
    """
    Packing an element of result returned by map matcher into a geojson
    """
    function linestr2coordinates(s::String)
        """
        Transform the linestring road returned by map matcher into coordinates
        coordinates = linestr2coordinates(routes[2])

        Input: "LINESTRING (126.647 45.783, 126.648 45.783)"
        Output: [[126.648, 45.784], [126.648, 45.7833]]
        """
        startswith(s, "LINESTRING") || return []
        pairs = split(s[13:end-1], ",")
        map(x -> parse.(Float64, split(x)), pairs)
    end

    function packjson(properties::Dict, coordinates::Array)
        """
        packjson(Dict("road"=>123), coordinates)
        """
        Dict("type"=>"Feature",
             "properties"=>properties,
             "geometry"=>Dict("type"=>"LineString",
                              "coordinates"=>coordinates))
    end

    road = get(dict, "road", -1)
    routes = get(dict, "route", "")
    coordinates = linestr2coordinates(routes)
    packjson(Dict("road"=>road), coordinates)
end

function matched2geojson(dicts::Vector)
    """
    Packing the result returned by map matcher into a geojson
    """
    features = matched2geojson.(dicts)
    Dict("type"=>"FeatureCollection",
         "crs"=>Dict("type"=>"name",
                     "properties"=>Dict("name"=>"urn:ogc:def:crs:OGC:1.3:CRS84")),
         "features"=>features)
end

function matched2dataframe(dicts::Vector)
    roads = map(d->get(d, "road", -1), dicts)
    fracs = map(d->get(d, "frac", ""), dicts)
    headings = map(d->get(d, "heading", ""), dicts)
    DataFrame(road=roads, frac=fracs, heading=headings)
end
