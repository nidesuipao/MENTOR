import os.path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def get_reward(g, eg):
    if g[0] >= 0 and g[1] >= 0:
        return -np.linalg.norm(g - eg, ord=2)
    if g[0] <= 0 and g[1] >= 0:
        return -np.linalg.norm(g - [0, 0.3], ord=2) - 0.25
    if g[0] <= 0 and g[1] <= 0:
        return -np.linalg.norm(g - [-0.3, 0], ord=2) - 0.6
    if g[0] >= 0 and g[1] <= 0:
        return -np.linalg.norm(g - [0, -0.3], ord=2) - 1

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def reward_hotpic_plot(env, step, reward_model, distanc_model, reward_norm, args, alpha, k):
    seed = args.seed
    max_feedback = args.max_feedback
    x = np.arange(-0.5, 0.5, 0.01)
    # x_r = np.repeat(x, [10], axis=0)

    y = np.arange(-0.5, 0.5, 0.01)
    # y_r = np.repeat(y, [10], axis=1)
    z = []

    for i in np.arange(-0.5, 0.5, 0.01):
        line = []
        for j in np.arange(-0.5, 0.5, 0.01):
            # rewardmodel.r_hat(np.concatenate([meta_obs, g]))
            reward = reward_model.r_hat(np.array([0.4, -0.4, 0.25, 0.25, i, j]))
            reward = reward_norm.normalize(reward)[0]
            # print(reward)
            line.append(reward)
        z.append(line)

    z2 = []
    for i in np.arange(-0.5, 0.5, 0.01):
        line = []
        for j in np.arange(-0.5, 0.5, 0.01):
            # rewardmodel.r_hat(np.concatenate([meta_obs, g]))
            inputx = np.array([0.4, -0.4, i, j])
            inputx = torch.tensor(inputx, dtype=torch.float32).cuda()
            distance = distanc_model(inputx)
            distance = distance.detach().cpu().item()
            # print(distance)
            line.append(distance)
        z2.append(line)

    # print("./figures/" + str(seed) + "_" + str(max_feedback))
    # print(os.path.exists("./figures/" + str(seed) + "_" + str(max_feedback)))
    if not os.path.exists("./figures/" + str(seed) + "_" + str(max_feedback)):
        os.mkdir("./figures/" + str(seed) + "_" + str(max_feedback) + "/")
    if not os.path.exists("./figures/" + str(seed) + "_" + str(max_feedback) + "/" + env):
        os.makedirs("./figures/" + str(seed) + "_" + str(max_feedback) + "/" + env + "/reward")
        os.mkdir("./figures/" + str(seed) + "_" + str(max_feedback) + "/" + env + "/distance")
        os.mkdir("./figures/" + str(seed) + "_" + str(max_feedback) + "/" + env + "/r_d")

    z = np.array(z).T
    c = plt.pcolormesh(x, y, z, cmap='RdBu_r', shading='nearest')
    # if step == 0:
    plt.colorbar(c)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("./figures/" + str(seed) + "_" + str(max_feedback) + "/" + env + "/reward/" + str(step) + ".jpg")
    plt.clf()
    plt.close("all")

    z2 = np.array(z2).T
    c = plt.pcolormesh(x, y, z2, cmap='RdBu_r', shading='nearest')
    # if step == 0:
    plt.colorbar(c)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("./figures/" + str(seed) + "_" + str(max_feedback) + "/" + env + "/distance/" + str(step) + ".jpg")
    plt.clf()
    plt.close("all")

    z2 = z2 - k
    z2 = np.maximum(z2, 0)
    z = z - alpha.item() * z2

    c = plt.pcolormesh(x, y, z, cmap='RdBu_r', shading='nearest')
    # if step == 0:
    plt.colorbar(c)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("./figures/" + str(seed) + "_" + str(max_feedback) + "/" + env + "/r_d/" + str(step) + ".jpg")
    plt.clf()
    plt.close("all")


if __name__=='__main__':
    x = np.arange(-0.5, 0.51, 0.01)
    # x_r = np.repeat(x, [10], axis=0)

    y = np.arange(-0.5, 0.51, 0.01)
    # y_r = np.repeat(y, [10], axis=1)
    z = []
    for i in np.arange(-0.5, 0.51, 0.01):
        line = []
        for j in np.arange(-0.5, 0.51, 0.01):
            line.append(get_reward(np.array([i,j]), np.array([0.25, 0.25])))
        z.append(line)

    z = np.array(z).T
    c = plt.pcolormesh(x, y, z, cmap='RdBu_r', shading='nearest')
    # if step == 0:
    cbar = plt.colorbar(c)
    cbar.ax.tick_params(labelsize=18)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    # plt.xticks(np.linspace(-0.5, 0.5, num=10))
    # plt.yticks(np.linspace(-0.5, 0.5, num=10))
    plt.rcParams.update({'font.size': 18})
    plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.subplots_adjust(bottom=0.2)
    # plt.subplots_adjust(left=0.2)
    x_ticks = np.linspace(-0.5, 0.5, num=5)
    y_ticks = np.linspace(-0.5, 0.4, num=5)  # 使其与 x 轴的范围和刻度数量相匹配
    from matplotlib.ticker import FixedLocator
    from matplotlib.ticker import MaxNLocator

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    plt.tight_layout()
    env = 'pointmass_rooms'
    if not os.path.exists("./figures/" + env):
        os.mkdir("./figures/" + env + "/")
    plt.savefig("./figures/" + env + "/" + "original.pdf")
    plt.close()


