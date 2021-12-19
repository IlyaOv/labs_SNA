import vk_api
import time
import os
import math
import requests


def auth_handler():
    """ При двухфакторной аутентификации вызывается эта функция.
    """
    # Код двухфакторной аутентификации
    key = input("Enter authentication code: ")
    # Если: True ‐ сохранить, False ‐ не сохранять.
    remember_device = True
    return key, remember_device


login, password = 'логин', 'пароль'
vk_session = vk_api.VkApi(
    login, password,
    auth_handler=auth_handler
)
try:
    vk_session.auth()
except vk_api.AuthError as error_msg:
    print(error_msg)

tools = vk_api.VkTools(vk_session)

url = "https://vk.com/album-4086_283220248"
# Разбираем ссылку
album_id = url.split('/')[-1].split('_')[1]
owner_id = url.split('/')[-1].split('_')[0].replace('album', '')

response = tools.get_all("photos.getAlbums", 100, {'owner_id': owner_id, 'album_ids': album_id})
photos_count = response['items'][0]['size']
print(response)
print("photos_count: ", photos_count)

counter = 0 # текущий счетчик
prog = 0 # процент загруженных
breaked = 0 # не загружено из-за ошибки
time_now = time.time() # время старта

if not os.path.exists('saved'):
    os.mkdir('saved')
photo_folder = 'saved/album{0}_{1}'.format(owner_id, album_id)
if not os.path.exists(photo_folder):
    os.mkdir(photo_folder)
for j in range(math.ceil(photos_count / 1000)):
    photos = tools.get_all("photos.get", 100, {'owner_id': owner_id, 'album_id': album_id, "count": 1000, "offset": j*1000, "v": 5.95})['items']
    # print(photos)
    for photo in photos:
        print(photo["id"])
        counter += 1
        sizes = photo['sizes']
        s = photo['sizes'][0]
        value_x = 0
        value_y = 0
        for size in sizes: #выбираем самый большой размер
            if value_x < size['width']:
                value_x = size['width']
                s = size
            if value_y < size['height']:
                value_y = size['height']
                s = size
        print(s['url'])
        url_ = s['url']
        print('Загружаю фото № {} из {}. Прогресс: {} %'.format(counter, photos_count, prog))
        prog = round(100/photos_count*counter,2)
        try:
            r = requests.get(url_)
            with open(f"{photo_folder}/{photo['id']}.jpg", 'wb') as outfile:
                outfile.write(r.content)
        except Exception:
            print('Произошла ошибка, файл пропущен.')
            breaked += 1
            continue
time_for_dw = time.time() - time_now
print("\nВ очереди было {} файлов. Из них удачно загружено {} файлов, {} не удалось загрузить.")