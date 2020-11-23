import numpy as np

''' Notes:

pos = self.lander.position
vel = self.lander.linearVelocity
state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
vel.x*(VIEWPORT_W/SCALE/2)/FPS,
vel.y*(VIEWPORT_H/SCALE/2)/FPS,
self.lander.angle,
20.0*self.lander.angularVelocity/FPS,
1.0 if self.legs[0].ground_contact else 0.0,
1.0 if self.legs[1].ground_contact else 0.0
            ]
'''

class robot:
    def __init__(self):
        # self.all_state = {1:"Start", 2:"Tipping Left", 3: "Tipping Right"}
        # curr_state = 1
        self.toggle = [0.5, 0.0, 0.0]
        # self.balance = 0
        self.throttle = 0
        # self.j = 0

    # clockwise = negative
    # counterclockwise = positive
    # First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power.
    # Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.
    def update(self, status, user_action):
        angular_vel = status[5]
        angle = status[4]
        # y_vel = status[]
        action = [0.0, 0.0]

        print(angle)

        if abs(angle) <= 0.3:
            if user_action == "left":
                action = [0.5, -1.0]
            elif user_action == "right":
                action = [0.5, 1.0]
            elif user_action == "up":
                action = [1.0, 0.0]
            elif user_action == "down":
                action = [0.0, 0.0]
        elif angle < -0.3:
            action = [0.5+abs(1/angle), -0.75+angle]
        elif angle > 0.3:
            action = [0.5+abs(1/angle), 0.75+angle]
        #
        # if not self.balance:
        #     if not self.throttle:
        #         action = [0.5, angular_vel*15]
        #     else:
        #         action = [0.0, angular_vel*15]
        # else:
        #     if user_action == "left":
        #         action = [0.5, -1.0]
        #     elif user_action == "right":
        #         action = [0.5, 1.0]
        #     elif user_action == "up":
        #         action = [0.75, angular_vel*15]
        #     elif user_action == "down":
        #         action = [0.0, angular_vel*15]
        #
        # self.balance = (self.balance + 1) % 5
        self.throttle = (self.throttle + 1) % 2

        return self.balance(status)

    # x = 0 center
    # y = 1.5 top
    def balance(self, status):
        x_pos = status[0]
        y_pos = status[1]
        x_vel = status[2]
        y_vel = status[3]
        angle = status[4]
        angular_vel = status[5]

        action = [0.0, 0.0]
        print("(x,y): ("+str(x_pos)+", "+str(y_pos)+")")

        if y_pos < 1.0:
            action[0] = 0.5 + (1.0-y_pos)
        else:
            action[0] = 0.0

        if x_pos < 0.0:
            action[1] = 0.5 + abs(x_pos)
        else:
            action[1] = -0.5 - abs(x_pos)

        # action[1] = angular_vel*20

        return action
