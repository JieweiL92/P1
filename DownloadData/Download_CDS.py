import cdsapi, os
from datetime import datetime, timedelta
from multiprocessing import Pool
from functools import partial
#
# # from 2017-01-01 to
# def DownloadAllERA5v(path):
#     c = cdsapi.Client()
#     c.retrieve(
#         'reanalysis-era5-single-levels',
#         {
#             'product_type': 'reanalysis',
#             'format': 'netcdf',
#             'variable': [
#                 '10m_v_component_of_wind'
#             ],
#             'year': [
#                 '2017', '2018', '2019'
#             ],
#             'month': [
#                 '01', '02', '03',
#                 '04', '05', '06',
#                 '07', '08', '09',
#                 '10', '11', '12'
#             ],
#             'day': [
#                 '01', '02', '03',
#                 '04', '05', '06',
#                 '07', '08', '09',
#                 '10', '11', '12',
#                 '13', '14', '15',
#                 '16', '17', '18',
#                 '19', '20', '21',
#                 '22', '23', '24',
#                 '25', '26', '27',
#                 '28', '29', '30',
#                 '31'
#             ],
#             'time': [
#                 '14:00', '15:00'
#             ]
#         },
#         path + 'vdownload.nc')
#     return None
#
#
#
# def DownloadAllERA5u(path):
#     c = cdsapi.Client()
#     c.retrieve(
#         'reanalysis-era5-single-levels',
#         {
#             'product_type': 'reanalysis',
#             'format': 'netcdf',
#             'variable': [
#                 '10m_u_component_of_wind'
#             ],
#             'year': [
#                 '2017', '2018', '2019'
#             ],
#             'month': [
#                 '01', '02', '03',
#                 '04', '05', '06',
#                 '07', '08', '09',
#                 '10', '11', '12'
#             ],
#             'day': [
#                 '01', '02', '03',
#                 '04', '05', '06',
#                 '07', '08', '09',
#                 '10', '11', '12',
#                 '13', '14', '15',
#                 '16', '17', '18',
#                 '19', '20', '21',
#                 '22', '23', '24',
#                 '25', '26', '27',
#                 '28', '29', '30',
#                 '31'
#             ],
#             'time': [
#                 '14:00', '15:00'
#             ]
#         },
#         path + 'udownload.nc')
#     return None

def DownloadERA5(datelist, path='F:/Jiewei/CDS/'):
    name = datetime.strftime(datelist, '%Y%m%d')
    file_list = os.listdir(path)
    ToF = True
    for file in file_list:
        if file.find(name)>=0:
            ToF = False
    if ToF == False:
        return None
    else:
        c = cdsapi.Client()
        yy = name[:4]
        mm = name[4:6]
        dd = name[6:]
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind'
                ],
                'year': yy,
                'month': mm,
                'day': dd,
                'time': [
                    '14:00', '15:00'
                ]
            },
            path + name + '.nc')
        return None


def Date_List():
    s1 = datetime(2017,1,11)
    s2 = datetime(2019,5,25)
    datelist = [s1]
    nows = s1
    while nows != s2:
        nows = nows + timedelta(days=12)
        datelist.append(nows)
    return datelist

if __name__ == '__main__':
    root = 'F:/Jiewei/CDS/'
    dl = Date_List()
    po = Pool()
    po.map(DownloadERA5, dl)
    po.close()
    po.join()
    # DownloadERA5(dl, root)
    # DownloadAllERA5u(root)
    # DownloadAllERA5v(root)