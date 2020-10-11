import numpy as np
import pygame
from matplotlib import pyplot as plt
import random
from math import sqrt


A1 = np.array([[np.random.uniform(0, 20), np.random.uniform(0, 20)] for k in range(8)])
A2 = np.array([[1, 1], [3, 8], [4, 2]])


def dist(list1, list2):
    return sum((i - j)**2 for i, j in zip(list1, list2))


dist = np.array([[dist(i, j) for i in A2] for j in A1])

# параметр неопределенности
m = 1.1
u = (1 / dist) ** (2 / (m - 1))
u = u / u.sum(axis=1)[:, None]
print(u)

# пересчитанные центры кластеров
(u.T).dot(A1) / u.sum(axis=0)[:, None]
l1 = u.max(axis=1)
l2 = u.argmax(axis=1)
l3 = np.array([i + 1 for i in l2])
l4 = np.array([i if j > .9 else 0 for i, j in zip(l3, l1)])

dataset = np.empty((0, 2), dtype='f')


def create_data(position):
    (x, y) = position
    r = np.random.uniform(0, 30)
    phi = np.random.uniform(0, 2 * np.pi)
    coord = [x + r * np.cos(phi), y + r * np.sin(phi)]
    global dataset
    dataset = np.append(dataset, [coord], axis=0)


radius = 2
color = (0, 0, 255)
thickness = 0

bg_color = (255, 255, 255)
(width, height) = (640, 400)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('C Means')

running = True
pushing = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pushing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            pushing = False

    if pushing and np.random.randint(0, 10) > 8:
        create_data(pygame.mouse.get_pos())
        pushing = False

    screen.fill(bg_color)

    for data in dataset:
        pygame.draw.circle(screen, color, (int(data[0]), int(data[2])), radius, thickness)

    pygame.display.flip()

pygame.quit()

dataset.shape


l1 = np.array([[1, 2], [6, 3], [5, 5], [2, 4], [9, 8], [1, 8], [4, 3], [8, 7]])


class CMeans:
    def __init__(self, dataset, n_clusters=3, m=1.2, k=.9):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.m = m
        self.k = k
        self.max_n_iter = 100
        self.tolerance = .01
        self.dist = np.zeros((self.dataset.shape[0], self.n_clusters))
        self.centroids = np.zeros((self.n_clusters, self.dataset.shape[1]))
        self.u = np.array([np.random.uniform(0, 1) for i in range(self.n_clusters)] for j in range(self.dataset.shape[0]))
        print(self.centroids)

    def distribute_data(self):
        self.dist = np.array([[self.get_dist2(i, j) for i in self.centroids] for j in self.dataset])
        self.u = (1 / self.dist) ** (2 / (self.m - 1))
        self.u = self.u / self.u.sum(axis=1)[:, None]

    def recalculate_centroids(self):
        self.centroids = (self.u.T).dot(self.dataset) / self.u.sum(axis=0)[:, None]

    def fit(self):
        iter = 1
        while iter < self.max_n_iter:
            prev_centroids = np.copy(self.centroids)
            self.recalculate_centroids()
            self.distribute_data()
            if max([self.get_dist2(i, k) for i, k in zip(self.centroids, prev_centroids)]) < self.tolerance:
                break
            iter += 1

    def get_labels(self):
        l1 = self.u.max(axis=1)
        l2 = self.u.argmax(axis=1)
        l3 = np.array([i + 1 for i in l2])
        return np.array([i if j > self.k else 0 for i, j in zip(l3, l1)])

    def get_dist2(self, list1, list2):
        return sum((i - j) ** 2 for i, j in zip(list1, list2))


test = CMeans(dataset, 3, 1.2, .9)
test.fit()
pred = test.get_labels()
print(pred)
colors = np.array(['#000000', '#377eb8', '#ff7f00', '#4daf4a'])
plt.figure()
plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[pred])
plt.show()
