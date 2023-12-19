import cv2
import numpy as np


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


vis_image = cv2.imread('/data/ckh/Object-Goal-Navigation/results/dump/exp1/episodes/thread_0/eps_1/0-1-Vis-0.png')
vis_image = np.ones_like(vis_image, dtype=np.uint8)
vis_image = np.ones((800, 800, 3), dtype=np.uint8)
print(vis_image.shape)
# vis_image = np.random.randint(0, 256, size=(480, 480), dtype=np.uint8) * 255
# agent_arrow = np.array([
#     [902, 284],
#     [909, 296],
#     [882, 284],
#     [909, 273]
# ])

agent_arrow = get_contour_points(pos=(400, 400, np.pi / 2), origin=(0, 0))
print(agent_arrow)

cv2.drawContours(vis_image, [agent_arrow], 0, (0, 255, 0), -1) # draw agent arrow
cv2.imshow("TEST", vis_image)
cv2.waitKey(0)