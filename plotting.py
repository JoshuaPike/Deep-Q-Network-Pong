import numpy as np
import math, random
import csv
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import statistics as st


def plotLoss(filename, fignum):
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=",")
    rows = list(reader)
    # 0 is loss vals
    # 2 is frame_idx for loss
    # 4 is reward val
    # 6 is reward frame_idx
    
    loss_vals = [float(x) for x in rows[0]]
    loss_frame = [float(x) for x in rows[2]]
    
    maxLoss = max(loss_vals)
    maxIdx = loss_vals.index(maxLoss)
    frameOfMax = loss_frame[maxIdx]
    print('Max Loss: ' + str(maxLoss) + ' at frame #: ' + str(frameOfMax))

    # try mean of every 1000 frames
    # print(str(st.fmean(loss_vals)))

    chunkSize = 5000

    loss_chunks = []
    loss_frame_chunks = []
    for x in range(0, len(loss_vals), int(chunkSize/4)):
        if x+chunkSize < len(loss_vals):
            loss_chunks.append(loss_vals[x:x+chunkSize])
            loss_frame_chunks.append(loss_frame[x:x+chunkSize])
    # loss_chunks = [loss_vals[x:x+chunkSize] for x in range(0, len(loss_vals), chunkSize)]
    # loss_frame_chunks = [loss_frame[x:x+chunkSize] for x in range(0, len(loss_frame), chunkSize)]

    mean_loss = [st.fmean(loss_chunks[x]) for x in range(0, len(loss_chunks))]
    frame_points = [st.fmean(loss_frame_chunks[x]) for x in range(0, len(loss_frame_chunks))]

    # The max is referring to mean loss
    maxLossMean = max(mean_loss)
    maxIdxMean = mean_loss.index(maxLossMean)
    frameOfMaxMean = frame_points[maxIdxMean]
    print('Max Mean Loss: ' + str(maxLossMean) + ' at frame #: ' + str(frameOfMaxMean))


    plt.figure(fignum)
    plt.title('$\gamma = 0.99$ | initial replay size = 15000 | buffer size = 80000')
    plt.plot(frame_points, mean_loss, linewidth = 1)
    plt.xlabel('Frame #')
    plt.ylabel('Loss')
    plt.grid()
    
    file.close()

def plotReward(filename, fignum):
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=",")
    rows = list(reader)
    reward_vals = [float(x) for x in rows[4]]
    reward_frame = [float(x) for x in rows[6]]

    # print(str(len(reward_vals)))
    last10 = reward_vals[-10:]
    average10 = st.mean(last10)
    print('Average reward of last 10 frames: ' + str(average10))

    chunkSize = 10

    reward_chunks = []
    reward_frame_chunks = []
    for x in range(0, len(reward_vals), int(chunkSize)):
        if x+chunkSize < len(reward_vals):
            reward_chunks.append(reward_vals[x:x+chunkSize])
            reward_frame_chunks.append(reward_frame[x:x+chunkSize])

    mean_reward = [st.fmean(reward_chunks[x]) for x in range(0, len(reward_chunks))]
    frame_points = [st.fmean(reward_frame_chunks[x]) for x in range(0, len(reward_frame_chunks))]



    plt.figure(fignum)
    plt.title('$\gamma = 0.99$ | initial replay size = 15000 | buffer size = 80000')
    plt.plot(frame_points, mean_reward, linewidth=1)
    plt.xlabel('Frame #')
    plt.ylabel('Reward')
    plt.grid()

    file.close()


num = 1
# Plot for file 1
# file1= 'Data/gamma_0.8_initial_5000_size_50000_numFrames_1000000.csv'
# plotLoss(file1, num)
# num+=1
# plotReward(file1, num)
# num+=1

# file2 = 'Data/gamma_0.66_initial_10000_size_100000_numFrames_1000000.csv'
# plotLoss(file2, num)
# num+=1
# plotReward(file2, num)
# num+=1

default = 'Data/gamma_0.99_initial_10000_size_100000_numFrames_1500000.csv'
# plotLoss(default, num)
# num+=1
# # plotLoss2(default, num)
# plotReward(default, num)
# num+=1

# Original best
ogBest = 'Data/gamma_0.99_initial_15000_size_100000_numFrames_1500000.csv'
# plotLoss(ogBest, num)
# num+=1
# plotReward(ogBest, num)
# num+=1

# file5 = 'Data/gamma_0.8_initial_10000_size_50000_numFrames_1000000.csv'
# plotLoss(file5, num)
# num+=1
# plotReward(file5, num)
# num+=1

# file6 = 'Data/gamma_0.33_initial_10000_size_100000_numFrames_1000000.csv'
# plotLoss(file6, num)
# num+=1
# plotReward(file6, num)
# num+=1


# file7 ='Data/gamma_0.99_initial_15000_size_150000_numFrames_1000000.csv'
# plotLoss(file7, num)
# num+=1
# plotReward(file7, num)
# num+=1


# file8 = 'Data/gamma_0.99_initial_15000_size_200000_numFrames_1000000.csv'
# plotLoss(file8, num)
# num+=1
# plotReward(file8, num)
# num+=1


file9 = 'Data/gamma_0.99_initial_50000_size_100000_numFrames_1000000.csv'
# plotLoss(file9, num)
# num+=1
# plotReward(file9, num)
# num+=1


# file10 = 'Data/gamma_0.99_initial_50000_size_250000_numFrames_1000000.csv'
# plotLoss(file10, num)
# num+=1
# plotReward(file10, num)
# num+=1

best = 'Data/gamma_0.99_initial_50000_size_100000_numFrames_2000000.csv'
# plotLoss(best, num)
# num+=1
# plotReward(best, num)
# num+=1


# converge = 'Data/gamma_0.99_initial_15000_size_80000_numFrames_1000000.csv'
# plotLoss(converge, num)
# num+=1
# plotReward(converge, num)
# num+=1

last = 'Data/gamma_0.99_initial_10000_size_80000_numFrames_1000000.csv'
plotLoss(last, num)
num+=1
plotReward(last, num)
num+=1
plt.show()
# Models/gamma_0.99_initial_15000_size_100000_numFrames_1500000.pth
# Models/gamma_0.99_initial_50000_size_100000_numFrames_2000000.pth


