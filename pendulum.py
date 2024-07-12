import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import sys
sys.path.append('.')

def make_gif(folder, gif_name, duration):
    '''
    Make a gif of the pendulum.
    '''
    images = []
    for i in range(1, len(os.listdir(folder))):
        images.append(imageio.imread(os.path.join(folder, f"{i}.png")))

    imageio.mimsave(f'{gif_name}.gif', images, duration= duration, loop = 0)

def draw_unit_circle(ax):
    '''
    Draw the unit circle.
    '''
    # draw the unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, linestyle='--', color='gray')

    # set the aspect ratio of the plot to be equal
    ax.set_aspect('equal', 'box')

    # set the limits of the plot
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

def draw_pendulum(ax, phi, color, label=None):
    '''
    Draw the pendulum given the angle w.r.t steady state.
    '''

    # pendulum length
    L = 1.0

    # pendulum position
    x = L * np.sin(phi)
    y = -L * np.cos(phi)

    # plot
    ax.plot([0, x], [0, y], color=color, lw=1, marker='o', markersize=10, label=label)
    ax.axis('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

def potential_energy(phi):
    '''
    Calculate the potential energy of the pendulum.
    '''
    # gravitational acceleration
    g = 9.81

    # potential energy
    return g * (1 - np.cos(phi))

def kinetic_energy(dphi):
    '''
    Calculate the kinetic energy of the pendulum.
    '''
    # kinetic energy
    return 0.5 * dphi**2


# In my case, where the length is 1, the equation rediuces to:
# \ddot{\theta} = -g \sin(\theta)
# This is not easy to solve (actually super hardcore stuff). We can use numerical methods to solve this.
# To do this, we need to convert this 2nd order ODE to a system of 1st order ODEs.

def pendulum_equation(t, z):
    '''
    Define the pendulum equation.

    return: [dz/dt, d^2z/dt^2]
    '''
    # gravitational acceleration
    g = 9.81

    # z[0] is the angle of the pendulum
    # z[1] is the angular velocity of the pendulum
    # The key point here is that z[1] is the integral of the second term
    # and the second term is the derivative of the first term.
    return [z[1], -g * np.sin(z[0])] # Here first term is \dot{\theta} and second term is \ddot{\theta}

def solve_pendulum(z0 = [np.pi/4, 0], derivative_system = pendulum_equation, t_range = (0, 10), evals = 200):
    '''
    Solve the pendulum equation.
    '''
    # This is kind of interesting, the fact that in the initial conditions I can specify the
    # initial angular velocity, that is more energy in the system actually...

    # time span
    t = np.linspace(t_range[0], t_range[1], evals)

    # solve the pendulum equation
    sol = solve_ivp(derivative_system, t_range, z0, t_eval=t)
    
    return sol # the first dimension contains \theta and the second dimension contains \dot{\theta}

def plot_phase_space(ax):
    '''
    Plot the phase space of the pendulum.
    '''
    # We can actually display the vector field in
    # velocity vs angle. This is because we have a definition 
    # of the change of valocity as a function of the angle.

    # Generate a grid of angle and velocity values
    angle = np.linspace(-np.pi, np.pi, 20)
    velocity = np.linspace(-20, 20, 20)
    A, V = np.meshgrid(angle, velocity)

    # Now, note that we are **defining** the velocity. We do not have
    # an explicit formula for it but it does not matter.

    dt = 1e-3
    # To calculate the direction:
    d_angle = lambda angle, velocity: angle + dt * velocity
    d_velocity = lambda angle, velocity: velocity + dt * (-9.81 * np.sin(angle))

    ax.set_xlabel('Angle')
    ax.set_ylabel('Velocity')
    ax.set_title('Phase space of the pendulum')
    
    # calculate the kinetic energy
    KE = kinetic_energy(V)
    # normalize the kinetic energy
    KE = KE / np.max(KE)
    # And now just plot the arrows
    q = ax.quiver(A, V, d_angle(A, V) - A, d_velocity(A, V) - V, KE, scale=0.3, linewidth=0.0001, headlength=3, headwidth=3)
    # Express the angle with pi using latex notation
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['$-\pi$', '$\\frac{-\pi}{2}$', '0', '$\\frac{\pi}{2}$', '$\pi$'])
    cbar = plt.colorbar(q, ax=ax)
    cbar.set_label('Normalized kinetic energy')

# Now, let's make a gif with two columns. In the first one, we will have the pendulum
# and in the second one, we will have the phase space. In the phase space we draw a point
# that represents the current state of the pendulum.
def dumped_pendulum_equation(t, z):
    '''
    Define the dumped pendulum equation.
    '''
    # gravitational acceleration
    g = 9.81

    # dumping coefficient
    c = 0.1

    # Note that sin(z[0]) will be negative in the lower part of the plane, which is where speed is indeed increasing.
    return [z[1], -g * np.sin(z[0]) - c * z[1]] # Here first term is \dot{\theta} and second term is \ddot{\theta}

def make_plots():
    #Generate an axis
    fig = plt.figure()
    ax = fig.add_subplot(111)

    draw_unit_circle(ax)
    draw_pendulum(ax, 0, 'black', "Steady state")
    draw_pendulum(ax, np.pi/4, 'red', "Perturbation")
    ax.legend()
    ax.set_title(f"{np.round(np.pi/4, 2)} rad perturbation")
    plt.savefig("pendulum1.png")
    plt.close()
    #GIF 1, evolution of potential energy

    os.makedirs('GIF1', exist_ok=True)
    n_images = 20
    for i in range(1, n_images):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        draw_unit_circle(ax)
        draw_pendulum(ax, 0, 'black', "Steady state")
        draw_pendulum(ax, np.pi * i/n_images, 'red')
        ax.set_title(f"{np.round(np.pi * i/n_images, 2)} rad perturbation \n Potential energy: {np.round(potential_energy(np.pi * i/n_images), 2)}")
        ax.legend()
        plt.savefig(f"GIF1/{i}.png")
        plt.close()
    make_gif('GIF1', 'pendulum1', 100)


    os.makedirs('GIF2', exist_ok=True)
    sol = solve_pendulum(z0 = [0.85*np.pi, 0]) # This is more fun changing the initial conditions

    for i in range(1, len(sol.t)):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # draw the pendulum
        draw_unit_circle(ax1)
        draw_pendulum(ax1, sol.y[0, i], 'black', 'Pendulum')
        ax1.set_title(f"Angle: {np.round(sol.y[0, i], 2)} rad")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # plot the phase space
        plot_phase_space(ax2)
        ax2.plot(np.sign(sol.y[0, i]) * (abs(sol.y[0, i]) % (np.pi)), sol.y[1, i], 'ro')
        ax2.set_title(f"Angle: {np.round(sol.y[0, i], 2)} rad \n Velocity: {np.round(sol.y[1, i], 2)} rad/s")
        ax2.set_xlabel('Angle')
        ax2.set_ylabel('Velocity')

        plt.savefig(f"GIF2/{i}.png")
        plt.close()

    make_gif('GIF2', 'pendulum2', 100)

    # Visualize the angle
    # As above we correct to have the angle between -2\pi and 2\pi. 
    plt.plot(sol.t, np.sign(sol.y[0]) * (abs(sol.y[0]) % (np.pi)))
    plt.title("Angle vs time")
    plt.xlabel("Time")
    plt.ylabel("Angle")
    plt.savefig("Angle_vs_time.png")


    os.makedirs('GIF4', exist_ok=True)
    sol = solve_pendulum(z0 = [0.85*np.pi, 0], derivative_system=dumped_pendulum_equation) # This is more fun changing the initial conditions

    for i in range(1, len(sol.t)):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # draw the pendulum
        draw_unit_circle(ax1)
        draw_pendulum(ax1, sol.y[0, i], 'black', 'Pendulum')
        ax1.set_title(f"Angle: {np.round(sol.y[0, i], 2)} rad")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # plot the phase space
        plot_phase_space(ax2)
        ax2.plot(np.sign(sol.y[0, i]) * (abs(sol.y[0, i]) % (np.pi)), sol.y[1, i], 'ro')
        ax2.set_title(f"Angle: {np.round(sol.y[0, i], 2)} rad \n Velocity: {np.round(sol.y[1, i], 2)} rad/s")
        ax2.set_xlabel('Angle')
        ax2.set_ylabel('Velocity')

        plt.savefig(f"GIF4/{i}.png")
        plt.close()

    make_gif('GIF4', 'pendulum4', 100)
    plt.close()
    plt.plot(sol.t, np.sign(sol.y[0]) * (abs(sol.y[0]) % (np.pi)))
    plt.title("Angle vs time")
    plt.xlabel("Time")
    plt.ylabel("Angle")
    plt.savefig("Angle_vs_time_dumped.png")


    ### Combined plot NN with numerical solution
    
    from Koopman import Koopman_autoencoder


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    my_k = Koopman_autoencoder(2, 64, 2)
    my_k.load_state_dict(torch.load('K_autoencoder.pth'))
    my_k = torch.compile(my_k)
    my_k.to(device)
    my_k.eval()

    theta_nn = []
    theta_dot_nn = []
    for t in np.linspace(0, 10, 200):
        pred_sol = my_k.predict(t, 0.85*np.pi, 0, device)
        theta_nn.append(pred_sol[0])
        theta_dot_nn.append(pred_sol[1])

    
    for i in range(1, len(sol.t)):
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # draw the pendulum
        draw_unit_circle(ax1)
        draw_pendulum(ax1, sol.y[0, i], 'red', 'Pendulum')
        draw_pendulum(ax1, theta_nn[i], 'blue', 'NN')
        ax1.set_title(f"Angle num. sol.: {np.round(sol.y[0, i], 2)} rad \n Angle NN: {np.round(theta_nn[i], 2)} rad")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # plot the phase space
        plot_phase_space(ax2)
        ax2.plot(np.sign(sol.y[0, i]) * (abs(sol.y[0, i]) % (np.pi)), sol.y[1, i], 'ro', label='Numerical solution')
        ax2.plot(np.sign(theta_nn[i]) * (abs(theta_nn[i]) % (np.pi)), theta_dot_nn[i], 'bo', label='NN solution')
        ax2.legend()
        ax2.set_title(f"Angle num. sol.: {np.round(sol.y[0, i], 2)} rad \n Velocity num. sol.: {np.round(sol.y[1, i], 2)} rad/s \n Angle NN: {np.round(theta_nn[i], 2)} rad \n Velocity NN: {np.round(theta_dot_nn[i], 2)} rad/s")
        ax2.set_xlabel('Angle')
        ax2.set_ylabel('Velocity')

        # Add legend indicating the numerical solution and the NN solution


        plt.savefig(f"GIF5/{i}.png")
        plt.close()

    make_gif('GIF5', 'pendulum5', 100)
    


if __name__ == "__main__":
    make_plots()