using DataFrames, CSV, HDF5

function read_cleandata(filename)
    df = CSV.read(filename)
    df = df[[:devid, :latitude, :longitude, :gpstime, :speed]]
    dateformat = DateFormat("y-m-d H:M:S")
    df[:gpstime] = DateTime(df[:gpstime], dateformat)
    df
end


function gpspoints2trips(df::DataFrame, mingap::Int, minduration::Int)
    """
    Split a device data into multiple trips if the time gap of two consecutive
    points is greater than mingap and only return trips whose duration are
    at least minduration.

    Input:
    mingap: minimum time gap (second)
    minduration: minimum duration time (minute)
    ---
    Return:
    An array of dataframes and each dataframe is a trip
    """
    ## devid blacklist
    BLACKLIST = Set([100000000])

    mingap = Dates.Second(mingap)
    minduration = Dates.Minute(minduration)
    trips = DataFrame[]
    function push_trip!(trips, g)
        if g[end, :gpstime] - g[1,:gpstime] >= minduration
            push!(trips, g)
        end
    end
    for g in groupby(df, [:devid])
        ## convert from SubDataFrame to DataFrame
        g = g[:,:]
        if g[1, :devid] in BLACKLIST
            continue
        end
        sort!(g, cols=[:gpstime])
        inds = Int64[]
        for row in eachrow(g)
            if length(inds) == 0
                push!(inds, row.row)
            else
                lastrow = g[inds[end], :]
                gap = row[:gpstime] - lastrow[:gpstime]
                if gap[1] < mingap
                    push!(inds, row.row)
                else
                    push_trip!(trips, g[inds, :])
                    inds = [row.row]
                end
            end
        end
        push_trip!(trips, g[inds, :])
    end
    trips
end
