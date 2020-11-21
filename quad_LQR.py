import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import control
from math import sin as s, cos as c


def plot_states():
    fig, axs = plt.subplots(2)
    fig.suptitle('Quadcopter States')
    axs[0].plot(time, quad.state_hist[:, 0], label='Roll')
    axs[0].plot(time, quad.state_hist[:, 1], label='Pitch')
    axs[0].plot(time, quad.state_hist[:, 2], label='Yaw')
    axs[0].plot(time, quad.state_hist[:, 3], label='Roll Rate')
    axs[0].plot(time, quad.state_hist[:, 4], label='Pitch Rate')
    axs[0].plot(time, quad.state_hist[:, 5], label='Yaw Rate')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(time, quad.state_hist[:, 6], label='X Vel')
    axs[1].plot(time, quad.state_hist[:, 7], label='Y Vel')
    axs[1].plot(time, quad.state_hist[:, 8], label='Z Vel')
    axs[1].plot(time, quad.state_hist[:, 9], label='X Pos')
    axs[1].plot(time, quad.state_hist[:, 10], label='Y Pos')
    axs[1].plot(time, quad.state_hist[:, 11], label='Z Pos')
    axs[1].grid()
    axs[1].legend()

    plt.show()


def plot_trajectory():
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.plot3D(x_pos, y_pos, z_pos, label='Quadcopter')
    ax.plot3D(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'gx', label='Waypoints')
    ax.legend()
    plt.title('Inertial Coordinates')
    plt.show()


def animated_plot():
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim(-5., 5.)
    ax.set_ylim(-5., 5.)
    ax.set_zlim(0., 10.)

    time_text = ax.text(4., 4., 12., 'Time')
    z_text = ax.text(4., 4., 11., 'Z')
    times = np.linspace(0., t_final, int(t_final/DT)+1)

    writer = animation.writers['ffmpeg']
    writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    def update(i):
        time_text.set_text("Time: {0:0.2f}".format(times[i]))
        z_text.set_text("Z: {0:0.2f}".format(round(z_pos[i], 2)))
        quad_pos.set_data(x_pos[i], y_pos[i])
        quad_pos.set_3d_properties(z_pos[i])
        waypt_pos.set_data(waypoints[:, 0], waypoints[:, 1])
        waypt_pos.set_3d_properties(waypoints[:, 2])
        return (quad_pos,)

    quad_pos, = plt.plot([], [], 'bX', markersize=10.)
    waypt_pos, = plt.plot([], [], 'go', markersize=10.)
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    line_ani = animation.FuncAnimation(fig, update, nsim, interval=1, repeat=True)
    plt.show()


class Quadcopter:
    def __init__(self, **init_kwargs):
        self.Ix = .0196
        self.Iy = .0196
        self.Iz = .0264

        self.g = 9.8  # m/s^2
        self.m = .5

        #  States
        self.x_dot = init_kwargs['x_dot'] if 'x_dot' in init_kwargs.keys() else 0.
        self.y_dot = init_kwargs['y_dot'] if 'y_dot' in init_kwargs.keys() else 0.
        self.z_dot = init_kwargs['z_dot'] if 'z_dot' in init_kwargs.keys() else 0.
        self.x = init_kwargs['x'] if 'x' in init_kwargs.keys() else 0.
        self.y = init_kwargs['y'] if 'y' in init_kwargs.keys() else 0.
        self.z = init_kwargs['z'] if 'z' in init_kwargs.keys() else 0.

        self.roll_dot = init_kwargs['roll_dot'] if 'roll_dot' in init_kwargs.keys() else 0.
        self.pitch_dot = init_kwargs['pitch_dot'] if 'pitch_dot' in init_kwargs.keys() else 0.
        self.yaw_dot = init_kwargs['yaw_dot'] if 'yaw_dot' in init_kwargs.keys() else 0.
        self.roll = init_kwargs['roll'] if 'roll' in init_kwargs.keys() else 0.
        self.pitch = init_kwargs['pitch'] if 'pitch' in init_kwargs.keys() else 0.
        self.yaw = init_kwargs['yaw'] if 'yaw' in init_kwargs.keys() else 0.

        self.states = np.array(
            [self.roll, self.pitch, self.yaw, self.roll_dot, self.pitch_dot, self.yaw_dot, self.x_dot, self.y_dot,
             self.z_dot, self.x, self.y, self.z]).T

        self.state_hist = self.states.copy()

        self.K, _, _ = control.lqr(self.A, self.B, self.Q, self.R)  # LQR Gain

    @property
    def A(self):
        A = np.zeros((12, 12))
        A[0, 3] = 1.
        A[1, 4] = 1.
        A[2, 5] = 1.
        A[6, 1] = -self.g
        A[7, 0] = self.g
        A[9, 6] = 1.
        A[10, 7] = 1.
        A[11, 8] = 1.
        return A

    @property
    def B(self):
        B = np.zeros((12, 4))
        B[3, 1] = 1/self.Ix
        B[4, 2] = 1/self.Iy
        B[5, 3] = 1/self.Iz
        B[8, 0] = 1/self.m
        return B

    @property
    def Q(self):
        Q = np.eye(12)
        # Weight position errors slightly more than other state errors
        Q[9, 9] = 3
        Q[10, 10] = 3
        Q[11, 11] = 5
        return Q

    @property
    def R(self):
        R = np.eye(4)*.1
        return R

    @property
    def actuator_effort(self):
        return self.control_array.copy()

    def record_states(self):
        self.state_hist = np.vstack((self.state_hist, self.states.reshape((1, -1))))

    def lqr_controller(self, desired_states):
        cmd = -self.K @ (self.states-desired_states)
        return cmd

    def update_states(self, ft, tx, ty, tz):
        roll_ddot = ((self.Iy - self.Iz) / self.Ix) * (self.pitch_dot * self.yaw_dot) + tx / self.Ix
        pitch_ddot = ((self.Iz - self.Ix) / self.Iy) * (self.roll_dot * self.yaw_dot) + ty / self.Iy
        yaw_ddot = ((self.Ix - self.Iy) / self.Iz) * (self.roll_dot * self.pitch_dot) + tz / self.Iz
        x_ddot = -(ft/self.m) * (s(self.roll) * s(self.yaw) + c(self.roll)*c(self.yaw) * s(self.pitch))
        y_ddot = -(ft/self.m) * (c(self.roll) * s(self.yaw) * s(self.pitch) - c(self.yaw) * s(self.roll))
        z_ddot = -1*(self.g - (ft/self.m) * (c(self.roll) * c(self.pitch)))

        self.roll_dot += roll_ddot*DT
        self.roll += self.roll_dot*DT
        self.pitch_dot += pitch_ddot * DT
        self.pitch += self.pitch_dot * DT
        self.yaw_dot += yaw_ddot * DT
        self.yaw += self.yaw_dot * DT

        self.x_dot += x_ddot * DT
        self.x += self.x_dot * DT
        self.y_dot += y_ddot * DT
        self.y += self.y_dot * DT
        self.z_dot += z_ddot * DT
        self.z += self.z_dot * DT

        self.states = np.array(
            [self.roll, self.pitch, self.yaw, self.roll_dot, self.pitch_dot, self.yaw_dot, self.x_dot, self.y_dot,
             self.z_dot, self.x, self.y, self.z]).T

        self.record_states()

    def __call__(self, ft=0., tx=0., ty=0., tz=0., xd=np.zeros(12)):
        cmds = self.lqr_controller(xd)
        thrust_cmd = cmds[0, 0]
        tx_cmd = cmds[0, 1]
        ty_cmd = cmds[0, 2]
        tz_cmd = cmds[0, 3]
        self.update_states(ft+thrust_cmd, tx_cmd, ty_cmd, tz_cmd)
        self.control_array = np.array([ft+thrust_cmd, tx_cmd, ty_cmd, tz_cmd])


if __name__ == "__main__":
    DT = .01
    init_dict = {'x_dot': .25, 'y_dot': .25, 'z_dot': .25, 'x': 0., 'y': 0., 'z': 0., 'roll_dot': .3, 'pitch_dot': .2,
                 'yaw_dot': .35, 'roll': .15, 'pitch:': .25, 'yaw': .2}
    quad = Quadcopter(**init_dict)

    x_pos = []
    y_pos = []
    z_pos = []
    time = []
    t = 0.
    t_final = 15.
    nsim = 0
    waypoints = np.array([[3., 5., 6.],
                          [-2., -4., 4.],
                          [-3., 5., 8.]])
    waypt_ctr = 0
    waypt_thresh = .15
    
    while t <= t_final:
        x_pos.append(quad.x)
        y_pos.append(quad.y)
        z_pos.append(quad.z)
        time.append(t)

        if np.linalg.norm(np.array([quad.x, quad.y, quad.z]) - waypoints[waypt_ctr, :]) <= waypt_thresh:
            print('WAYPOINT {%i} reached' % waypt_ctr)
            waypt_ctr += 1
        if waypt_ctr >= len(waypoints):
            print('MISSION FINISHED')
            break

        x_ref = np.zeros(12)
        x_ref[-3:] = waypoints[waypt_ctr]
        hover_thrust = quad.m*quad.g
        quad(hover_thrust, xd=x_ref)
        t += DT

        nsim += 1

    print('FINAL POS: ', round(quad.x, 2), round(quad.y, 2), round(quad.z, 2))
    plot_states()
    plot_trajectory()
    animated_plot()
