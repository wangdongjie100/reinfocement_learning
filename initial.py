# !apt install python-opengl
# !apt install ffmpeg
# !apt install xvfb
# !pip3 install pyvirtualdisplay

import gym
from gym import wrappers
# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

env = gym.make("SpaceInvaders-v0")
env = wrappers.Monitor(env, "./tmp/SpaceInvaders-v0", force=True)

for episode in range(2):
    observation = env.reset()
    step = 0
    total_reward = 0

    while True:
        step += 1
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode: {0},\tSteps: {1},\tscore: {2}"
                  .format(episode, step, total_reward)
            )
            break
env.close()