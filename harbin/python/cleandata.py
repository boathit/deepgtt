
## This file reads the original csv files in datapath/0*/*.csv, drop duplicates
## saves the clean data into datapath/cleandata/*.csv
## cleandata: 1,632,649,039 points

import pandas as pd
import os

MIN_LAT, MIN_LON = 45.657920, 126.506130
MAX_LAT, MAX_LON = 45.830905, 126.771862
cols = ['DEVID', 'LATITUDE', 'LONGTITUDE', 'GPSTIME', 'SPEED', 'STATE', 'ALARMFLAG', 'ORIENTATION']
## (devid,latitude,longitude,gpstime,speed,state,alarmflag,orientation)
def reduce_redundancy(df):
    df = df[cols]
    df = df[(df.LATITUDE >= MIN_LAT) & (df.LATITUDE <= MAX_LAT) &
            (df.LONGTITUDE >= MIN_LON) & (df.LONGTITUDE <= MAX_LON)]
    n = df.shape[0]
    df = df.drop_duplicates(subset=['DEVID', 'LATITUDE', 'LONGTITUDE'])
    df.columns = ['devid', 'latitude', 'longitude', 'gpstime', 'speed', 'state', 'alarmflag', 'orientation']
    #print("{} reduces to {}".format(n, df.shape[0]))
    return df

#import sqlite3, psycopg2
#from sqlalchemy import create_engine
# def createdb_sqlite():
#     con = sqlite3.connect("/home/xiucheng/data-backup/bigtable/harbin-taxi.db")
#     con.execute('''create table gpspoints (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                                            devid INT,
#                                            latitude DOUBLE,
#                                            longitude DOUBLE,
#                                            gpstime ,
#                                            speed INT,
#                                            state INT,
#                                            alarmflag INT,
#                                            orientation INT)''')
#     con.commit()
#     con.close()
# def write2db_sqlite(df):
#     con = sqlite3.connect("/home/xiucheng/data-backup/bigtable/harbin-taxi.db")
#     pd.io.sql.to_sql(df, con=con, name='gpspoints', index=False, if_exists='append', flavor='sqlite')
#     con.close()
# def write2db_pg(df):
#     con = psycopg2.connect("dbname='harbin-taxi' user='postgres' host='localhost' port='5431' password='123456'")
#     con.execute('''''')
#     con.close()

if __name__ == "__main__":
    datapath = "/home/xiucheng/data-backup/bigtable/2015-taxi-csv/data/"
    savepath = os.path.join(datapath, "cleandata")
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    monthdirs = ["01", "02", "03", "04", "05"]
    numpoints = 0
    for monthdir in monthdirs:
        absmonthdir = os.path.join(datapath, monthdir)
        csvfiles = filter(lambda s: s.endswith("csv"), os.listdir(absmonthdir))
        csvrealpaths = map(lambda s: os.path.join(absmonthdir, s), sorted(csvfiles))
        for fname in csvrealpaths:
            df = pd.read_csv(fname)
            print("{} nrows0: {}".format(os.path.basename(fname), df.shape[0]))
            df = reduce_redundancy(df)
            print("{} nrows1: {}".format(os.path.basename(fname), df.shape[0]))
            print("saving to csv...")
            df.to_csv(os.path.join(savepath, os.path.basename(fname)), index=False)
            numpoints += df.shape[0]
    print("Total number of points inserted: {}".format(numpoints))
