
import psycopg2
import geopandas as gpd
import pandas as pd
import numpy as np
import json

host = "localhost"
port = 5432
database = "harbin"
user     = "osmuser"
password = "pass"

con = psycopg2.connect(host=host, port=port,
                       database=database, user=user,
                       password=password)
con.set_client_encoding("UTF8")

bfmap_ways = gpd.read_postgis("select gid, osm_id, source, target, reverse, priority, geom from bfmap_ways;", con)
ways = pd.read_sql_query("select id as osm_id, tags, nodes from ways;", con)

roads = bfmap_ways.merge(ways, on='osm_id')
tags = roads.tags.apply(lambda s: json.loads('{'+s.replace('=>', ':')+'}'))


def get_lengths():
    gid = bfmap_ways.gid.values
    lengths = bfmap_ways.length.values * 1000
    return np.r_[0., lengths]

def get_dict_u():
    gid = roads.gid.values
    dict_u = {}
    dict_u[0] = 0
    for road in gid:
        dict_u[road] = road
    return dict_u, len(dict_u)

def get_dict_s1():
    """
    s1: highway
    """
    highways = tags.apply(lambda d: d.get("highway", "unclassified"))
    types = np.sort(highways.unique())
    type_id = {}
    for i in range(len(types)):
        type_id[types[i]] = i
    df = pd.concat([roads.gid, highways.apply(lambda t: type_id.get(t))],
                   axis=1, keys=['gid', 'type_id'])
    dict_s1 = {}
    dict_s1[0] = 0
    for _, row in df.iterrows():
        dict_s1[row.gid] = row.type_id
    return dict_s1, len(highways.unique())

def get_dict_s2():
    """
    s2: number of lanes
    """
    lanes = tags.apply(lambda d: d.get("lanes", "0"))
    df = pd.concat([roads.gid, lanes.apply(lambda x: int(x))],
                   axis=1, keys=['gid', 'lane'])
    dict_s2 = {}
    dict_s2[0] = 0
    for _, row in df.iterrows():
        dict_s2[row.gid] = row.lane
    return dict_s2, len(lanes.unique())

def get_dict_s3():
    """
    s3: one way or not
    """
    oneway = tags.apply(lambda d: d.get("oneway", "0"))
    oneway = oneway.apply(lambda s: s if s == "0" else "1")
    df = pd.concat([roads.gid, oneway.apply(lambda x: int(x))],
                   axis=1, keys=['gid', 'oneway'])
    dict_s3 = {}
    dict_s3[0] = 0
    for _, row in df.iterrows():
        dict_s3[row.gid] = row.oneway
    return dict_s3, len(oneway.unique())
