from datetime import datetime, timedelta
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import math
import Download.Sentinel_Data as ds


footprint = ''      # WKT format
orbit_circle = 1    # in 175
download_path = 'F:/Jiewei/Sentinel-1/Level1-GRD-IW'





def DateList(d):
    First_day = datetime(2018,1,1).date()
    Last_day = datetime.now().date()
    origin_time = d
    if hasattr(d, 'date'):
        origin_time= d.date()
    time_step = timedelta(days=12)

    n = math.ceil((origin_time - First_day).days/12)
    L1 = [n-i-1 for i in range(n)]
    d1 = [origin_time-time_step*i for i in L1]
    n = math.ceil((Last_day - origin_time).days/12)
    L1 = [i for i in range(n)]
    d2 = [origin_time+time_step*i for i in L1]
    d1.extend(d2[1:])
    DL = [i for i in d1 if First_day<=i<=Last_day]
    return DL


def DownloadData(products, api):
    Off_List = []
    print(len(products))
    n = 0
    for t in products:
        info = api.get_product_odata(t.uuid)
        n = n+1
        print(n)
        if info['Online']:
            print('Product {} is online. Starting download.'.format(t.uuid))
            api.download(t.uuid, directory_path = download_path)
        else:
            print('Product {} is not online.'.format(t.uuid))
            Off_List.append(t.uuid)
    return Off_List



if __name__ == '__main__':
    t1 = datetime(2019,5,1)
    t = DateList(t1)
    for s in t:
        print(s)
