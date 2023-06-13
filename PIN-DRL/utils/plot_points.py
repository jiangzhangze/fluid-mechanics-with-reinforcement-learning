import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from fluid_mechanics.area import *
fg, ax = plt.subplots()
cylinder = Circle(xy=(0.3, 0.2), radius=0.05, fill=False, color="black")
ax.add_patch(cylinder)
domain = Rectangle(xy=(0, 0), width=2.2, height=0.41, fill=False, color="black")
ax.add_patch(domain)
ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
points1 = probes(x_min=0.4, y_min=0.05, length=0.3, num=100)
points2 = circle_probes(0.05, 4)# 4
points3 = circle_probes(0.07, 20)
points4 = add_points(points1, points2) # 104
points5 = add_points(points2, points3) # 24
points = add_points(points4, points3) # 124
x = points5[0]
y = points5[1]
plt.scatter(x, y, s=5)
plt.axis('equal')
plt.show()
