import vk_api
import time
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

user1 = 314692966


def auth_handler():
    """ При двухфакторной аутентификации вызывается эта функция.
    """
    # Код двухфакторной аутентификации
    key = input("Enter authentication code: ")
    # Если: True ‐ сохранить, False ‐ не сохранять.
    remember_device = True
    return key, remember_device


def stop_f(items):
    print(items)


def get_groups_users(friends_list, tools):
    friends_out = {}
    for friend in friends_list:
        try:
            friends_out[friend] = tools.get_all('friends.get', 100, {'user_id': friend})
        except Exception:
            friends_out[friend] = []
        time.sleep(1)
    return friends_out


def main():
    login, password = 'логин', 'пароль'
    vk_session = vk_api.VkApi(
        login, password,
        auth_handler=auth_handler  # функция для обработки двухфакторной аутентификации
    )
    try:
        vk_session.auth()
    except vk_api.AuthError as error_msg:
        print(error_msg)

    tools = vk_api.VkTools(vk_session)
    friend_list = []
    friend_list.append(user1)

    # friends_out = get_groups_users(friend_list, tools)
    #
    # with open('friends_out.pkl', 'wb') as output:
    #     pickle.dump(friends_out, output, pickle.HIGHEST_PROTOCOL)

    with open('friends_out.pkl', 'rb') as input:
        friends_out = pickle.load(input)

    print(friends_out)

    # friends_friends = get_groups_users(friends_out[user1]['items'], tools)
    #
    # with open('friends_friends.pkl', 'wb') as output:
    #     pickle.dump(friends_friends, output, pickle.HIGHEST_PROTOCOL)

    with open('friends_friends.pkl', 'rb') as input:
        friends_friends = pickle.load(input)

    print(friends_friends)

    g = make_graph(friends_out, friends_friends)
    plot_graph(g, 500)


# def make_graph(friends_out, friends_friends):
#     graph = nx.Graph()
#     graph.add_node(user1, size=friends_out[user1]['count'])
#     for i in friends_out[user1]['items']:
#         try:
#             graph.add_node(i, size=friends_friends[i]['count'])
#             intersection = set(friends_out[user1]['items']).intersection(set(friends_friends[i]['items']))
#             graph.add_edge(user1, i, weight=len(intersection))
#         except Exception:
#             print("err")
#     print(graph)
#     print(graph.nodes, "\n", graph.edges)
#     return graph

def make_graph(friends_out, friends_friends):
    graph = nx.Graph()
    graph.add_node(user1, size = friends_out[user1]['count'])
    for i in friends_out[user1]['items']:
        try:
            graph.add_node(i, size = friends_friends[i]['count'])
            intersection = set(friends_out[user1]['items']).intersection(set(friends_friends[i]['items']))
            graph.add_edge(user1, i, weight=len(intersection))
        except Exception:
            print("err")
    for i in range(len(friends_out[user1]['items'])):
        id1= friends_out[user1]['items'][i]
        for k in range(i+1, len(friends_out[user1]['items'])):
            id2= friends_out[user1]['items'][k]
            try:
                intersection = set(friends_friends[id1]['items']).intersection(set(friends_friends[id2]['items']))
                if len(intersection) > 0:
                    graph.add_edge(id1, id2, weight=len(intersection))
            except Exception:
                print("err friend")
    print(graph)
    print(graph.nodes, "\n", graph.edges)
    return graph


def plot_graph(graph, adjust_nodesize):
    # pos = nx.drawing.layout.circular_layout(graph)
    pos = nx.spring_layout(graph, k=0.1)
    # нормализуем размер вершины для визуализации. Оптимальное значение параметра
    # adjust_nodesize ‐ от 300 до 500

    nodesize = [graph.nodes[i]['size'] / adjust_nodesize for i in graph.nodes()]
    # нормализуем толщину ребра графа. Здесь хорошо подходит
    # нормализация по Standard Score

    edge_mean = np.mean([graph.edges[i]['weight'] for i in graph.edges()])
    edge_std_dev = np.std([graph.edges[i]['weight'] for i in graph.edges()])
    edgewidth = [((graph.edges[i]['weight'] - edge_mean) / edge_std_dev / 2) for i in graph.edges()]

    # создаем граф для визуализации
    nx.draw_networkx_nodes(graph, pos, node_size=nodesize, node_color='y', alpha=0.9)
    nx.draw_networkx_edges(graph, pos, width=edgewidth, edge_color = 'b')
    nx.draw_networkx_labels(graph, pos)
    # сохраняем и показываем визуализированный граф
    plt.savefig('saved')
    plt.show()


if __name__ == '__main__':
    main()