from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 1000000
batch_size = 32
gamma = 0.99
record_idx = 10000

# gamma_list = [0.99]
# replay_initial_list = [15000]
# buffer_size_list = [120000]

# gamma_list = [0.99]
# replay_initial_list = [15000]
# buffer_size_list = [80000]

gamma_list = [0.99]
replay_initial_list = [50000]
buffer_size_list = [100000]

# replay_initial = 10000
# replay_buffer = ReplayBuffer(100000)

for gamma in gamma_list:
    for replay_initial in replay_initial_list:
        for buffer_size in buffer_size_list:
            nameOfFile = 'gamma_' + str(gamma) + '_initial_' + str(replay_initial) + '_size_' + str(buffer_size) + '_numFrames_' + str(num_frames)
            # for when batch_size != 32
            if batch_size != 32:
                 nameOfFile += '_batchSize_' + str(batch_size)


            # nameOfFile = 'gamma_' + str(gamma) + '_initial_' + str(replay_initial) + '_size_' + str(buffer_size) + '_numFrames_' + str(num_frames)
            # model_name = 'Models/gamma_' + str(gamma) + '_initial_' + str(replay_initial) + '_size_' + str(buffer_size) + '_numFrames_' + str(num_frames) + '.pth'
            model_name = 'Models/' + nameOfFile + '.pth'
            print('\ngamma: ' + str(gamma) + '   replay_initial: ' + str(replay_initial) + '   buffer_size: ' + str(buffer_size))

            replay_buffer = ReplayBuffer(buffer_size)
            model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
            model.load_state_dict(torch.load("model_pretrained.pth", map_location='cpu'))

            target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
            target_model.copy_from(model)

            optimizer = optim.Adam(model.parameters(), lr=0.00001)
            if USE_CUDA:
                model = model.cuda()
                target_model = target_model.cuda()
                print("Using cuda")

            epsilon_start = 1.0
            epsilon_final = 0.01
            epsilon_decay = 30000
            epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

            losses = []
            all_rewards = []
            episode_reward = 0

            state = env.reset()

            for frame_idx in range(1, num_frames + 1):
                #print("Frame: " + str(frame_idx))

                epsilon = epsilon_by_frame(frame_idx)
                action = model.act(state, epsilon)
    
                next_state, reward, done, _ = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
    
                state = next_state
                episode_reward += reward
    
                if done:
                    state = env.reset()
                    all_rewards.append((frame_idx, episode_reward))
                    episode_reward = 0

                if len(replay_buffer) > replay_initial:
                    loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append((frame_idx, loss.data.cpu().numpy()))

                if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
                    print('#Frame: %d, preparing replay buffer' % frame_idx)

                if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
                    print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses, 0)[1]))
                    print('Last-10 average reward: %f' % np.mean(all_rewards[-10:], 0)[1])

                if frame_idx % 50000 == 0:
                    target_model.copy_from(model)

            # Save model
            print("Saving model...")
            torch.save(model.state_dict(), model_name)

            # Write Data to file
            print("Writing data to file...")
            file_name = 'Data/' + nameOfFile + '.csv'
            # file_name = 'Data/gamma_' + str(gamma) + '_initial_' + str(replay_initial) + '_size_' + str(buffer_size) + '_numFrames_' + str(num_frames) + '.csv'
            file = open(file_name, 'w')
            writer = csv.writer(file)
            frame_losses, loss_vals = zip(*losses)
            writer.writerow(list(loss_vals))
            writer.writerow(frame_losses)

            frame_rewards, rewards = zip(*all_rewards)
            writer.writerow(rewards)
            writer.writerow(frame_rewards)
            file.close()

            # Plotting
            # print("Plotting...")
            # loss_graph = 'Images/LOSS_gamma_' + str(gamma) + '_initial_' + str(replay_initial) + '_size_' + str(buffer_size) + '_numFrames_' + str(num_frames) +'.png'
            # reward_graph = 'Images/REWARD_gamma_' + str(gamma) + '_initial_' + str(replay_initial) + '_size_' + str(buffer_size) + '_numFrames_' + str(num_frames) +'.png'

            # plt.figure(1)
            # plt.plot(frame_losses[0::100], list(loss_vals)[0::100], linewidth=1)
            # plt.xlabel('Frame #')
            # plt.ylabel('Loss')
            # plt.grid()
            # plt.savefig(loss_graph)

            # plt.figure(2)
            # plt.plot(frame_rewards, rewards, linewidth=1)
            # plt.xlabel('Frame #')
            # plt.ylabel('Reward')
            # plt.grid()
            # plt.savefig(reward_graph)



# for gamma in gamma_list:
#     for replay_initial in replay_initial_list:
#         for buffer_size in buffer_size_list:
#             replay_buffer = ReplayBuffer(buffer_size)

# model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
# model.load_state_dict(torch.load("model_pretrained.pth", map_location='cpu'))

# target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
# target_model.copy_from(model)

# optimizer = optim.Adam(model.parameters(), lr=0.00001)
# if USE_CUDA:
#     model = model.cuda()
#     target_model = target_model.cuda()
#     print("Using cuda")

# epsilon_start = 1.0
# epsilon_final = 0.01
# epsilon_decay = 30000
# epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# losses = []
# all_rewards = []
# episode_reward = 0

# state = env.reset()

# for frame_idx in range(1, num_frames + 1):
#     #print("Frame: " + str(frame_idx))

#     epsilon = epsilon_by_frame(frame_idx)
#     action = model.act(state, epsilon)
    
#     next_state, reward, done, _ = env.step(action)
#     replay_buffer.push(state, action, reward, next_state, done)
    
#     state = next_state
#     episode_reward += reward
    
#     if done:
#         state = env.reset()
#         all_rewards.append((frame_idx, episode_reward))
#         episode_reward = 0

#     if len(replay_buffer) > replay_initial:
#         loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         losses.append((frame_idx, loss.data.cpu().numpy()))

#     if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
#         print('#Frame: %d, preparing replay buffer' % frame_idx)

#     if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
#         print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses, 0)[1]))
#         print('Last-10 average reward: %f' % np.mean(all_rewards[-10:], 0)[1])

#     if frame_idx % 50000 == 0:
#         target_model.copy_from(model)

# file_name = 'gamma_' + str(gamma) + '_initial_' + str(replay_initial) + '_size'_ + str(buffer_size) + '.csv'
# file = open(file_name, 'w')
# writer = csv.writer(file)
# writer.writerow(losses)
# writer.writerow(all_rewards)

# # Plotting
# loss_graph = 'LOSS_gamma_' + str(gamma) + '_initial_' + str(replay_initial) + '_size'_ + str(buffer_size) + '.png'
# reward_graph = 'REWARD_gamma_' + str(gamma) + '_initial_' + str(replay_initial) + '_size'_ + str(buffer_size) + '.png'

# x = np.linspace(0, 1, num_frames)
# plt.figure(1)
# plt.plot(x, losses, linewidth=2)
# plt.xlabel('Frame #')
# plt.ylabel('Loss')
# plt.grid()
# plt.savefig(loss_graph)
# plt.show()

# plt.figure(2)
# plt.plot(x, all_rewards, linewidth=2)
# plt.xlabel('Frame #')
# plt.ylabel('Reward')
# plt.grid()
# plt.savefig(reward_graph)
# plt.show()
