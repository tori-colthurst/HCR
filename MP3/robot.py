import math

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
        self.x = 0.0
        self.y = 1.0
        self.avg_err_x = 0.0
        self.avg_err_y = 0.0
        self.avg_err_ang = 0.0
        self.count = 0

        self.toggle = 1
        self.step_size = 0.02

    # clockwise = negative
    # counterclockwise = positive
    # First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power.
    # Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.
    def update(self, status, user_action):

        x0 = status[0]
        y0 = status[1]
        x_vel = status[2]
        y_vel = status[3]
        angle = status[4]
        angular_vel = status[5]

        x_target = self.x
        y_target = self.y

        if user_action == "left": # positive
            x_target = self.x - self.step_size
            y_target = self.y
        elif user_action == "right":
            x_target = self.x + self.step_size
            y_target = self.y
        elif user_action == "up":
            x_target = self.x
            y_target = self.y + self.step_size
        elif user_action == "down":
            x_target = self.x
            y_target = self.y - self.step_size

        self.x = x_target
        self.y = y_target

        if self.toggle > 0:
            self.count_err(x0, y0, angle)
            self.toggle = -self.toggle

        b = self.balance(angle, angular_vel, y0, y_vel, x0, x_vel, x_target, y_target)
        return b

    # x = 0 center
    # y = 1.5 top
    def balance(self, ang, ang_v, y, y_vel, x, x_vel, x_target, y_target):
        # p = p_0 + v_0*t + 0.5*a*t^2
        # a = (p - p_0 - v_0) / (0.5)
        t = 0.5
        scale = 1.0

        a = (0.0 - (ang + ang_v*t)) / (0.5*t**2)
        x_acc = (x_target - (x + x_vel*t)) / (0.5*t**2)
        y_acc = (y_target - (y + y_vel*t)) / (0.5*t**2)

        if abs(ang) > math.pi/16:
            scale = 0.5

        left_right_engine = -a + x_acc
        main_engine = y_acc * scale

        action = [main_engine, left_right_engine]

        return action

    def get_error(self):

        self.avg_err_x /= self.count
        self.avg_err_y /= self.count
        self.avg_err_ang /= self.count

        print("AVG ERRORS: ")
        print("X: "+str(self.avg_err_x))
        print("Y: "+str(self.avg_err_y))
        print("ANG: "+str(self.avg_err_ang))

    def count_err(self, x0, y0, angle):

        self.avg_err_x += abs(self.x-x0)
        self.avg_err_y += abs(self.y-y0)
        self.avg_err_ang += abs(angle)

        self.count += 1
