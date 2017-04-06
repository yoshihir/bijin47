'''Getter images from bijin watch.'''
import os
import datetime
import urllib.request

AREAS = ['hokkaido',
         'aomori',
         'iwate',
         'sendai', # miyagi
         'akita',
         #'yamagata', nothing
         'fukushima',
         'ibaraki',
         'tochigi',
         'gunma',
         'saitama',
         'chiba',
         'tokyo',
         'kanagawa',
         'niigata',
         #'toyama', nothing
         'kanazawa',
         'fukui',
         'yamanashi',
         'nagano',
         #'gifu', nothing
         'shizuoka',
         'nagoya', #aichi
         #'mie', nothing
         #'shiga', nothing
         'kyoto',
         'osaka',
         'kobe', #hyogo
         'nara',
         #'wakayama', nothing
         'tottori',
         #'shimane', nothing
         'okayama',
         'hiroshima',
         'yamaguchi',
         'tokushima',
         'kagawa',
         #'ehime', nothing
         #'kochi', nothing
         'fukuoka',
         'saga',
         'nagasaki',
         'kumamoto',
         #'oita', nothing
         'miyazaki',
         'kagoshima',
         'okinawa',
        ]

def get_image():
    '''
    Getter images from bijin watch
    '''
    for area in AREAS:
        dirpath = './bijinwatch' + '/' + area + '/'
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        url = 'http://www.bijint.com/assets/pict/' + area + '/pc/'
        time_standard = datetime.datetime(2017, 1, 1, 0, 0)

        while True:
            file_name = time_standard.strftime('%H%M') + '.jpg'
            urllib.request.urlretrieve(url + file_name, dirpath + file_name)
            print(dirpath + file_name + ' output')

            time_standard += datetime.timedelta(minutes=1)
            if time_standard.strftime('%H%M') == '0000':
                break

if __name__ == '__main__':
    get_image()
