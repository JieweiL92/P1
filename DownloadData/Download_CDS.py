import cdsapi, os
import DownloadData.Download_sentinel as dds
from datetime import datetime
import netCDF4 as ncdf

def DownloadAllERA5(path):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind'
            ],
            'year': [
                '2017', '2018', '2019'
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12'
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31'
            ],
            'time': [
                '14:00', '15:00'
            ]
        },
        path + 'download.nc')
    return None


def DownloadERA5(datelist, path):
    c = cdsapi.Client()
    for dt in datelist:
        yy = str(dt.year)
        mm = str(dt.month)
        dd = str(dt.day)
        if len(mm) == 1:
            mm = '0'+mm
        if len(dd) == 1:
            dd = '0'+dd
        name = str(dt)
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

if __name__ == '__main__':
    root = 'F:/Jiewei/CDS/'
    dl = dds.DateList(datetime(2019,5,25))
    DownloadERA5(dl[0:3], root)

