import gym
import numpy as np
import os
import sys
import time
from argparse import ArgumentParser
import robot

# majority of code obtained from https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code

# ------------------------- Code for keyboard agent ------------------------- #
# from https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py

# Keyboard controls:
# w - Nop
# a - fire right engine
# s - fire main engine
# d - fire left engine

do_user_action = False
user_action = -1
main_engine = 0.0
left_right_engine = 0.0

def key_press(k, mod):
    global do_user_action, user_action
    if k == ord('w'):
        user_action = 0
        do_user_action = True
        main_engine = 0.0
        left_right_engine = 0.0
    if k == ord('a'):
        user_action = 3
        do_user_action = True
        main_engine = 0.0
        left_right_engine = 0.5

    if k == ord('s'):
        user_action = 2
        do_user_action = True
        main_engine = 0.5
        left_right_engine = 0.0

    if k == ord('d'):
        user_action = 1
        do_user_action = True
        main_engine = 0.0
        left_right_engine = -0.5

def key_release(k, mod):
    global do_user_action, user_action
    do_user_action = False
    user_action = -1
    main_engine = 0.0
    left_engine = 0.0
    right_engine = 0.0


# -------------------------------------------------------------------------- #

def main(args):
    env = gym.make('LunarLanderContinuous-v2')

    # enable key presses
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    global do_user_action, user_action, main_engine, left_right_engine

    for i in range(args.num_episodes):
        state = env.reset()
        total_reward = 0

        prev_action = None
        keypress_cntr = 0.0
        next_state = [0]*8

        r = robot.robot()
        action = [0.0, 0.0]

        while True:

            my_action = [0.0, 0.0]
            a = ""
            if do_user_action:

                # "acceleration" logic
                # the more number of time steps a key remains pressed, the power of engine is increased
                if user_action == 0: # up, w
                    a = "up"
                    my_action[0] = 0.0
                    my_action[1] = 0.0

                    prev_action = 0
                    keypress_cntr = 0.0

                elif user_action == 3: # left, a
                    a = "left"
                    my_action[0] = 0.0
                    my_action[1] = 0.5

                    if prev_action == 3:
                        keypress_cntr += 0.1

                    else:
                        prev_action = 3
                        keypress_cntr = 0.0

                    my_action[1] += keypress_cntr

                    if my_action[1] > 1.0:
                        my_action[1] = 1.0

                elif user_action == 2: # down, s
                    a = "down"
                    my_action[0] = 0.5
                    my_action[1] = 0.0

                    if prev_action == 2:
                        keypress_cntr += 0.1

                    else:
                        prev_action = 2
                        keypress_cntr = 0.0

                    my_action[0] += keypress_cntr

                    if my_action[0] > 1.0:
                        my_action[0] = 1.0

                elif user_action == 1: # right, d
                    a = "right"
                    my_action[0] = 0.0
                    my_action[1] = -0.5

                    if prev_action == 1:
                        keypress_cntr += -0.1

                    else:
                        prev_action = 1
                        keypress_cntr = 0.0

                    my_action[1] += keypress_cntr

                    if my_action[1] < -1.0:
                        my_action[1] = -1.0

                else:
                    my_action[0] = 0.0
                    my_action[1] = 0.0

                    prev_action = 0
                    keypress_cntr = 0.0

            action = r.update(next_state, a)
            next_state, reward, done, info = env.step(action)
            # print('next state: '+str(next_state))

            total_reward += reward
            env.render()
            if done:
                print('Episode', i, ': reward =', total_reward)
                break
            state = next_state
            time.sleep(0.05)

    env.close()
    r.get_error()


if __name__ == "__main__":
    parser = ArgumentParser(description='LunarLander-v2 Continuous')
    parser.add_argument('--num_episodes', type=int, default = 1,
                        help='number of episodes for training')
    args = parser.parse_args()
    main(args)
