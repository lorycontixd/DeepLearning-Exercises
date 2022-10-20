import gym

env = gym.make("MountainCar-v0", render_mode='rgb_array')
env.reset()

done = False
while not done:
    action = 2  # always go right!
    env.step(action)
    env.render()