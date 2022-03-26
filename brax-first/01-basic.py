from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np

import brax

# A boundy ball scene
bouncy_ball = brax.Config(dt=0.05, substeps=4)
ground = bouncy_ball.bodies.add(name='ground')
ground.frozen.all = True
plane = ground.colliders.add().plane
plane.SetInParent()

ball = bouncy_ball.bodies.add(name='ball', mass=1)
cap = ball.colliders.add().capsule
cap.radius, cap.length = 0.5, 1

bouncy_ball.gravity.z = -9.8


def draw_system(ax, pos, alpha=1):
    for i, p in enumerate(pos):
        ax.add_patch(Circle(xy=(p[0], p[2]), radius=cap.radius, fill=False, color=(0, 0, 0, alpha)))
        if i < len(pos) - 1:
            pn = pos[i + 1]
            ax.add_line(Line2D([p[0], pn[0]], [p[2], pn[2]], color=(1, 0, 0, alpha)))


def main():

    qp = brax.QP(
        # position of each body in 3d (z is up, right-hand coordinate)
        pos=np.array([[0., 0., 0.],  # ground
                      [0., 0., 3.]]),  # ball is 3m up in the air
        # velocity of each body in 3d
        vel=np.array([[0., 0., 0.],  # ground
                      [0., 0., 0.]]),  # ball
        # rotation about center of body, as a quaternion (w, x, y, z)
        rot=np.array([[1., 0., 0., 0.],  # ground
                      [1., 0., 0., 0.]]),  # ball
        # angular velocity about center of body in 3d
        ang=np.array([[0., 0., 0.],  # ground
                      [0., 0., 0.]])  # ball
    )

    bouncy_ball.elasticity = 0.75 #@param { type:"slider", min: 0, max: 0.95, step:0.05 }
    ball_velocity = 1 #@param { type:"slider", min:-5, max:5, step: 0.5 }

    sys = brax.System(bouncy_ball)

    # provide an initial velocity to the ball
    qp.vel[1, 0] = ball_velocity

    _, ax = plt.subplots()
    plt.xlim([-3, 3])
    plt.ylim([0, 4])

    for i in range(100):
      draw_system(ax, qp.pos[1:], i / 100.)
      qp, _ = sys.step(qp, [])

    plt.title('ball in motion')
    plt.show()


def basic_show():
    _, ax = plt.subplots()
    plt.xlim([-3, 3])
    plt.ylim([0, 4])

    draw_system(ax, [[0, 0, 0.5]])
    plt.title('ball at rest')
    plt.show()


if __name__ == '__main__':
    main()
