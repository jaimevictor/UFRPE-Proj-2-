import matplotlib.pyplot as plt
import numpy as np
import time

def format_input(q_line):
    points = {}
    for n in range(q_line):  # Recebe a quantidade correta de linhas
        line = list(map(str, input().split()))  # Põe tudo numa lista
        for j in range(len(line)):  # Percorre a linha pra adicionar no dicionario
            if line[j] != '0':
                points[line[j]] = (n, j)
    return points


def permutations(points):
    if len(points) == 1:
        return [points]
    final_list = []
    for index, m in enumerate(points):
        points_left = points[:index] + points[index+1:]
        z = permutations(points_left)
        for t in z:
            final_list.append([m] + t)
    return final_list


def routing(points):  # Permuta todos os pontos
    del points['R']  # Remove o inicio temporariamente da conta
    points = list(item for item in points)
    permutation = permutations(points)
    routes = ['R' + "".join(route) + 'R' for route in permutation]  # Adiciona ao fim e começo de cada rota permutada
    return routes


def calculate_route(route, allpoints):  # Calcula o custo da rota recebida
    cost = 0
    for k in range(len(route)-1):
        actual_point = route[k]
        next_point = route[k+1]
        actual_point = allpoints.get(actual_point)  # Recolhe os valores do ponto atual
        next_point = allpoints.get(next_point)  # recolhe os valores do prox ponto
        cost += (abs(actual_point[0]-next_point[0]) + abs(actual_point[1]-next_point[1]))  # Soma o custo dessa movimentação até a final
    return cost


def plot_route(route, allpoints):
    X = []
    Y = []
    route_size = len(route)-1
    for k in range(route_size):
        actual_point = route[k]
        next_point = route[k+1]
        actual_point = allpoints.get(actual_point)  # Recolhe os valores do ponto atual
        next_point = allpoints.get(next_point)  # recolhe os valores do prox ponto
        X.append(actual_point[1]) # adicionando ponto atual
        Y.append(actual_point[0]) # adicionando ponto atual
        X.append(actual_point[1]) # adicionando ponto de conexão
        Y.append(next_point[0]) # adicionando ponto de conexão
        if k == route_size:
            X.append(next_point[1]) # adicionando ponto de final
            Y.append(next_point[0]) # adicionando ponto de final

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def create_graph (route, all_points):

    x_axis, y_axis = plot_route(route, all_points) # The axis
    graph, (plot1) = plt.subplots(1) #
    plot1.plot(x_axis, y_axis, "-o") # the plot itself
    plot1.invert_yaxis()             #
    # beautify

    plot1.set_title("Best Route")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.grid()

    for b in all_points:
        point = all_points[b]
        x = point[1]
        y = point[0]
        plt.text(x, y, b, horizontalalignment='right')

    plt.show()


while True:
    q_line, q_column = list(map(int, input().split()))
    all_points = format_input(q_line)
    all_routes = routing(all_points.copy())
    route_cost = [0] * len(all_routes)
    for i in range(len(all_routes)):
        route = all_routes[i]
        route_cost[i] = calculate_route(route, all_points)
    lowest_cost = min(route_cost)  # Menor custo entre todas as rotas
    cheap_route = all_routes[route_cost.index(lowest_cost)]  # Pegar indice da lista do menor valor e recolher rota de mesmo indice
    print(cheap_route)
    create_graph(cheap_route, all_points)
