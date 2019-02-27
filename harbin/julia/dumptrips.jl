include("Trip.jl")

function dumptrips(infile::String, outfile::String)
    trips = readtrips(infile)
    trips = filter(t->t.tms[end]-t.tms[1]>=7*60, trips)
    h5open(outfile, "w") do f
        f["/meta/ntrips"] = length(trips)
        for i = 1:length(trips)
            f["/trip/$i/lon"] = trips[i].lon
            f["/trip/$i/lat"] = trips[i].lat
            f["/trip/$i/tms"] = trips[i].tms
            f["/trip/$i/devid"] = trips[i].devid
        end
    end
    println("Dumped $(length(trips)) trips.")
    length(trips)
end

inputpath = "/home/xiucheng/data-backup/bigtable/2015-taxi/h5path"
outputpath = "/home/xiucheng/data-backup/bigtable/2015-taxi/tmp"

fnames = readdir(inputpath)
n = 0
for fname in fnames
    infile = joinpath(inputpath, fname)
    outfile = joinpath(outputpath, fname)
    n += dumptrips(infile, outfile)
end
println("Total: $n trips.")
