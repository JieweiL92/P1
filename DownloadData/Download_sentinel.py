from datetime import datetime, timedelta
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import math
import Download.Sentinel_Data as ds
import Read_SentinelData.SentinelClass as rd


footprint = ''      # WKT format
orbit_circle = 1    # in 175
download_path = 'F:/Jiewei/Sentinel-1/Level1-GRD-IW/WhiteCity/'

westcoast = 'POLYGON((-124.32458920204549 46.43911834045062,-124.39597857281898 44.05214836263448,-124.89570416823345\
                        42.781768881082286,-124.32458920204549 41.32413925785548,-124.61014668513947 40.32477337122387,\
                        -124.00333703356475 39.613526657015825,-123.75347423585751 38.50489668171062,-122.39707619116108\
                         36.78092223486466,-120.61234192182363 33.92853172065848,-117.89954583243073 33.39375962613249,\
                         -117.14995743930902 33.542640657129155,-120.25539506795614 34.40108432451113,-123.07527521350929\
                          38.83931256315918,-124.28889451665874 42.65064046872291,-123.75347423585751 46.34063710349591,\
                          -124.32458920204549 46.43911834045062,-124.32458920204549 46.43911834045062))'
pg = 'POLYGON((-124.66368871321961 44.64877478932752,-124.26212350261868 44.59160969579332,-124.40490224416568 \
                    44.13866741936644,-124.95816986766029 44.1770796997578,-124.66368871321961 44.64877478932752,\
                    -124.66368871321961 44.64877478932752))'
WhiteCity = 'POLYGON((-125.18126165132745 41.97094443637533,-124.78862011207322 41.97094443637533,-124.78862011207322 42.36777641842906,-125.18126165132745 42.36777641842906,-125.18126165132745 41.97094443637533))'
AboveSantaRosa = 'POLYGON((-125.16341430863409 39.220574196715006,-124.44952060089912 39.220574196715006,-124.44952060089912 39.46901243508779,-125.16341430863409 39.46901243508779,-125.16341430863409 39.220574196715006))'
MontereyBay = 'POLYGON((-122.00443465190682 36.70583945034366,-121.93304528113332 36.70583945034366,-121.93304528113332 36.927300596433696,-122.00443465190682 36.927300596433696,-122.00443465190682 36.70583945034366))'
SantaBarbara = 'POLYGON((-120.4338684948899 34.2610714736297,-120.02337961294228 34.2610714736297,-120.02337961294228 34.39372105430317,-120.4338684948899 34.39372105430317,-120.4338684948899 34.2610714736297))'
LA = 'POLYGON((-118.78298929575276 33.86001462630601,-118.51081731967881 33.86001462630601,-118.51081731967881 33.9303827699067,-118.78298929575276 33.9303827699067,-118.78298929575276 33.86001462630601))'
SanDiego = 'POLYGON((-117.7835381049238 32.76942818514645,-117.56044632125665 32.76942818514645,-117.56044632125665 33.14381186553938,-117.7835381049238 33.14381186553938,-117.7835381049238 32.76942818514645))'

def DateList(d):
    First_day = datetime(2017,1,1).date()
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
            Off_List.append(t)
    return Off_List



if __name__ == '__main__':
    footprint = WhiteCity
    orbit_circle = 13
    # orbit_circle = 'None'
    s1ab = 'S1A*'
    product, api = ds.ExtractProducts(footprint, s1ab, orbit_circle)
    time_set = set()
    for t in product:
        time_set.add(t.time.date())
    time_estimated = set(DateList(product[0].time.date()))
    Missing_date = time_estimated - time_set
    print(Missing_date)
    Out = DownloadData(product, api)