import vk_api

def auth_handler():
    """ При двухфакторной аутентификации вызывается
    эта функция.
     """
    # Код двухфакторной аутентификации
    key = input("Enter authentication code: ")
    # Если: True - сохранить, False - не сохранять.
    remember_device = True
    return key, remember_device


def stop_f(items):
    print(items)


def main():
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
    vk_app = vk_session.get_api()
    print(vk_app.wall.post(message='Hello world!'))


if __name__ == '__main__':
    main()