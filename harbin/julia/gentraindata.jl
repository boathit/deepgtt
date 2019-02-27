using ArgParse

include("traindata.jl")

args = let s = ArgParseSettings()
    @add_arg_table s begin
        "--inputpath"
            arg_type=String
            default="/home/xiucheng/data-backup/bigtable/2015-taxi/data/jldpath"
        "--outputpath"
            arg_type=String
            default="/home/xiucheng/data-backup/bigtable/2015-taxi/data/trainpath"
    end
    parse_args(s; as_symbols=true)
end

###########################################################################
savetraindata(args[:outputpath], args[:inputpath])
