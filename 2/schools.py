import matplotlib.pyplot as plt
import numpy as np

with open('input.txt', 'r') as file:
    data = file.read().splitlines()

city = list(map(int, data[0].split()))
klen = city[0]
k_x = [klen * i for i in range(city[1] + 1)]
k_y = [klen * i for i in range(city[2] + 1)]
schoolnumber = int(data[1])
school = []
for i in range(2, 2 + schoolnumber):
    school.append(list(map(int, data[i].split())))
school = np.array(school)
pointnumber: int = int(data[2 + schoolnumber])
points = []
for i in range(3 + schoolnumber, 3 + schoolnumber + pointnumber):
    points.append(list(map(int, data[i].split())))
points = np.array(points)
k = 2


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    if (x1 // klen) == (x2 // klen) and (y1 // klen) != (y2 // klen):
        dy = abs(y1 - y2)
        dx = min((x1 % klen + x1 % klen), (2 * klen - (x1 % klen + x2 % klen)))
    elif (x1 // klen) != (x2 // klen) and (y1 // klen) != (y2 // klen):
        dx = abs(x1 - x2)
        dy = min((y1 % klen + y1 % klen), (2 * klen - (y1 % klen + y2 % klen)))
    else:
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

    return dx + dy


def closest_point(all_points, new_point):
    best_point = None
    best_distance = None

    for current_point in all_points:
        current_distance = distance(new_point, current_point)

        if best_distance is None or current_distance < best_distance:
            best_distance = current_distance
            best_point = current_point

    return best_point


def build_kdtree(points, depth=0):
    n = len(points)

    if n <= 0:
        return None
    axis = depth % k
    sorted_points = sorted(points, key=lambda point: point[axis])
    return {
        'point': sorted_points[n // 2],
        'left': build_kdtree(sorted_points[:n // 2], depth + 1),
        'right': build_kdtree(sorted_points[n // 2 + 1:], depth + 1)
    }


def closer_distance(pivot, p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    d1 = distance(pivot, p1)
    d2 = distance(pivot, p2)
    if d1 < d2:
        return p1
    else:
        return p2


def kdtree_closest_point(root, point, depth=0):
    if root is None:
        return None
    axis = depth % k
    next_branch = None
    opposite_branch = None
    if point[axis] < root['point'][axis]:
        next_branch = root['left']
        opposite_branch = root['right']
    else:
        next_branch = root['right']
        opposite_branch = root['left']
    best = closer_distance(point,
                           kdtree_closest_point(next_branch,
                                                point,
                                                depth + 1),
                           root['point'])
    if distance(point, best) > (point[axis] - root['point'][axis]) ** 2:
        best = closer_distance(point,
                               kdtree_closest_point(opposite_branch,
                                                    point,
                                                    depth + 1),
                               best)

    return best


tree = build_kdtree(school)

for p in points:
    print(p, kdtree_closest_point(tree, p))

fig, ax = plt.subplots()
ax.scatter(school[:, 0], school[:, 1])
ax.scatter(points[:, 0], points[:, 1])
ax.set_xticks(k_x)
ax.set_yticks(k_y)
plt.grid(True)
plt.show()