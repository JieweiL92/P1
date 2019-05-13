import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind'
        ],
        'year': '2017',
        'month': '01',
        'day': '01',
        'time': [
            '14:00', '15:00'
        ]
    },
    'download.grib')
