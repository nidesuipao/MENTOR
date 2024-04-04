import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


class Human_Label_Manager():
    def __init__(self, env, seed):

        self.fig = None
        self.ax = None
        self.pic_path = "HumanLabel/picture"
        self.label_path = "HumanLabel/picture"

        self.data_file = "HumanLabel/data.text"
        self.label_file = "label.txt"
        self.f = open(self.data_file, 'w+')
        self.f.close()

        self.env = env
        self.count = 0


    def background(self):
        fig, ax = plt.subplots(figsize=(10,10))

        start_point = [-0.55, 0]
        end_point = [-0.32, 0]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-')

        start_point = [-0.28, 0]
        end_point = [0.55, 0]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-')

        start_point = [0, -0.55]
        end_point = [0, -0.32]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-')

        start_point = [0, -0.28]
        end_point = [0, 0.28]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-')

        start_point = [0, 0.32]
        end_point = [0, 0.55]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k-')

        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.55, 0.55)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig("HumanLabel/background.pdf", pad_inches=0)


    def xy2pixxy(self, x, y):
        px = int((x+0.55)/1.1 * 311)
        py = int((0.55-y)/1.1 * 314)
        return (px,py)

    def get_reward(self, g, eg):

        if g[0] >= 0 and g[1] >= 0:
            return -np.linalg.norm(g - eg, ord=2)
        if g[0] <= 0 and g[1] >= 0:
            return -np.linalg.norm(g - [0, 0.3], ord=2) - 0.25
        if g[0] <= 0 and g[1] <= 0:
            return -np.linalg.norm(g - [-0.3, 0], ord=2) - 0.6
        if g[0] >= 0 and g[1] <= 0:
            return -np.linalg.norm(g - [0, -0.3], ord=2) - 1
        return -3

    def data_store_and_pic_gen(self, state1, state2):
        bgpic = cv2.imread("HumanLabel/background.jpg")
        # print(bgpic.shape)
        slen = len(state1)
        self.f = open(self.data_file, 'a+')
        labels = []
        for i in range(slen):
            store_pic = bgpic.copy()
            s1 = state1[i]
            s2 = state2[i]
            position1 = s1[:2]
            goal1 = s1[-2:]
            position2 = s2[:2]
            goal2 = s2[-2:]
            # print(position1)
            # print(self.xy2pixxy(position1[0], position1[1]))
            #
            # print(goal1)
            # print(self.xy2pixxy(goal1[0], goal1[1]))

            cv2.circle(store_pic, self.xy2pixxy(position1[0], position1[1]), 10, (255,0,0), thickness=2)
            cv2.circle(store_pic, self.xy2pixxy(goal1[0], goal1[1]), 10, (0, 255, 0), thickness=2)

            cv2.circle(store_pic, self.xy2pixxy(position2[0], position2[1]), 5, (0, 0, 0), thickness=-1)
            cv2.circle(store_pic, self.xy2pixxy(goal2[0], goal2[1]), 5, (128, 128, 128), thickness=-1)
            cv2.imwrite("HumanLabel/picture/" + str(self.count) + ".jpg", store_pic)
            datalen = str(
                [self.count, round(position1[0], 3), round(position1[1], 3), round(goal1[0], 3), round(goal1[1], 3),
                 round(position2[0], 3), round(position2[1], 3), round(goal2[0], 3),
                 round(goal2[1], 3)])

            r_1 = self.get_reward(goal1, [0.25, 0.25])
            r_2 = self.get_reward(goal2, [0.25, 0.25])
            reward_diff = round(r_1*1.0 - r_2*1.0, 3)
            rational_label = 1*(r_1 < r_2)
            cv2.imshow('Image 1', store_pic)
            key = cv2.waitKey(0)  # 等待键盘输入

            # 根据用户输入进行处理
            if key == ord('0'):
                rational_label = 0
            elif key == ord('1'):
                rational_label = 1

            labels.append(rational_label)

            datalen = datalen.replace('[','').replace(']', '') + ","+ str(reward_diff) +"," + str(rational_label) +'\n'
            self.f.write(str(datalen))
            self.count += 1

        self.f.close()
        cv2.destroyAllWindows()
        return np.array(labels)


x = Human_Label_Manager(123, 123)
x.background()
