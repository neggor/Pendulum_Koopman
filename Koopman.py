## Inspired by https://www.nature.com/articles/s41467-018-07210-0

import sys
sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn
from pendulum import solve_pendulum, pendulum_equation, dumped_pendulum_equation
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm



class encoder(nn.Module):
    '''
    Encoder for Koopman operator.
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
            )
        
    def forward(self, x):
        return self.encoder(x)
    
class decoder(nn.Module):
    '''
    Decoder for Koopman operator.
    '''
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
            )
        
    def forward(self, z):
        return self.decoder(z)

class K(nn.Module):
    '''
    Koopman operator.
    '''
    def __init__(self, latent_dim, hidden_dim):
        super(K, self).__init__()
        # Explicitly parameterize oscillaroty behaviour of the pendulum
        self.auxiliary_network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
            )
    
    def Jordan_block(self, magnitude, angle):
        '''
        Generate a Jordan block.
        '''
        magnitude = magnitude.unsqueeze(1).unsqueeze(1)
        angle = angle.unsqueeze(1).unsqueeze(1)
        
        # Create rotation matrix for each sample in the batch
        rotation = torch.cat([torch.cos(angle), -torch.sin(angle),
                            torch.sin(angle), torch.cos(angle)], dim=1)
        rotation = rotation.view(-1, 2, 2)  # Reshape to (batch_size, 2, 2)
    
        # Multiply magnitude by rotation matrix
        jordan_block = torch.exp(magnitude) * rotation

        return jordan_block
    
    def forward(self, z, dt):
        '''     
        Forward pass of the Koopman operator.
        '''
        # Get the parameters for the auxiliary network
        aux = self.auxiliary_network(z)
        dt = dt.unsqueeze(1)
        aux = aux * dt # Evaluate eigenvalues at time dt from previous time step
        
        # Generate K as a Jordan block
        block_1 = self.Jordan_block(aux[:, 0], aux[:, 1])
        #block_2 = self.Jordan_block(aux[:, 2], aux[:, 3])
        #identity_block = torch.eye(2).unsqueeze(0).repeat(z.shape[0], 1, 1)
        #K = torch.zeros(z.shape[0], 4, 4)
        #K[:, :2, :2] = block_1
        #K[:, 2:, 2:] = block_2
        #K[:, :2, 2:] = identity_block
        K = block_1.to(z.device)
       
        z = z.unsqueeze(2)
        x_hat = torch.matmul(K, z).squeeze(2)
        return x_hat

class Koopman_autoencoder(nn.Module):
    '''
    Koopman autoencoder.
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Koopman_autoencoder
        , self).__init__()
        self.encoder = encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = decoder(latent_dim, hidden_dim, input_dim)
        self.K = K(latent_dim, hidden_dim)


    def forward(self, x, dt):
        z = self.encoder(x)
        z_next = self.K(z, dt)
        x_next = self.decoder(z_next)
        return x_next

    def predict(self, t, theta_0, theta_dot_0, device):
        t = torch.tensor(t, dtype=torch.float32)
        theta_0 = torch.tensor(theta_0, dtype=torch.float32)
        theta_dot_0 = torch.tensor(theta_dot_0, dtype=torch.float32)
        input = torch.tensor([t, theta_0, theta_dot_0], dtype=torch.float32).unsqueeze(0).to(device)
        return self.forward(input[:, 1:], input[:, 0]).cpu().detach().numpy()[0]

def intrinsic_reconstruction_loss(encoder, decoder, x):
    '''
    Intrinsic reconstruction loss.
    '''
    z = encoder(x)
    x_hat = decoder(z)
    loss = torch.mean((x - x_hat)**2)
    return loss

def linear_dynamics_loss(K_autoencoder, x, y):
    '''
    Linear dynamics loss.
    '''
    dt = x[:, 0]
    x_0 = x[:, 1:]
    x_next = K_autoencoder(x_0, dt) # Predict the state after dt
    loss = torch.mean((x_next - y)**2)
    return loss


def generate_data(n_samples, damped = False, t_range=(0, 2), evals=100):
    """
    The idea is to generate tuples of (t, theta_0, theta_dot_0, theta_t). 
    """

    initial_theta_range = [-np.pi, np.pi]
    initial_theta_dot_range = [-np.pi, np.pi] # This is change of position per unit of time. 
                                             # \pi velocity means that the pendulum goes from one extreme to the other in one unit of time.
    
    data = []
    for _ in range(n_samples):
        z0 = [np.random.uniform(*initial_theta_range), np.random.uniform(*initial_theta_dot_range)]
        if damped:
            sol = solve_pendulum(z0=z0, t_range=t_range, derivative_system=dumped_pendulum_equation, evals = evals)
        else:
            sol = solve_pendulum(z0=z0, t_range=t_range, evals = evals)
        for i in range(1, len(sol.t)):
            # t, theta_0, theta_dot_0, theta_t, theta_dot_t
            data.append((sol.t[i], z0[0], z0[1], sol.y[0, i], sol.y[1, i]))
    
    # suffle the data
    np.random.shuffle(data)
    # assert that \theta is between -\pi and \pi
    #assert all(-np.pi <= d[3] <= np.pi for d in data)
    # assert that the time is not above or below the range
    assert all(t_range[0] <= d[0] <= t_range[1] for d in data)
    return data

def generate_data_torch(n_samples, damped, device, batch_size=256, t_range=(0, 2), evals=200):
    data_list = generate_data(n_samples, damped, t_range=t_range, evals=evals)
    data = np.array(data_list)
    X = data[:, :-2]
    y = data[:, -2:]
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(epochs, K_autoencoder, optimizer, train_loader, device):
    '''
    Train the Koopman autoencoder
    '''
    K_autoencoder.train()
    loss_list = []
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = intrinsic_reconstruction_loss(K_autoencoder.encoder, K_autoencoder.decoder, y)
            loss += linear_dynamics_loss(K_autoencoder, x, y)
            loss.backward()
            optimizer.step()
            # save model
            torch.save(K_autoencoder.state_dict(), 'K_autoencoder.pth')
            loss_list.append(loss.item())
        print(f'Epoch {epoch + 1}, Loss: {np.mean(loss_list[-len(train_loader):]):.4f}')
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ## Train
    my_k = Koopman_autoencoder(2, 64, 2)
    optimizer = torch.optim.Adam(my_k.parameters(), lr=1e-4)
    my_k.to(device)
    #my_k.load_state_dict(torch.load('K_autoencoder.pth'))
    my_k = torch.compile(my_k)
    print(my_k)
    train_loader = generate_data_torch(5000, damped=True, device=device)
    train(256, my_k, optimizer, train_loader, device)
    t_interval = (0, 10)
    t_range = np.linspace(t_interval[0], t_interval[1], 100)
    initial_conditions = [
        (0, 0.5*np.pi, 0),
        (0, 0.25*np.pi, 3),
        (0, 0.75*np.pi, 1),
        (0, 0.9*np.pi, 0)
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, (_, theta_0, theta_dot_0) in enumerate(initial_conditions):
        theta_actual = solve_pendulum([theta_0, theta_dot_0], t_range= t_interval, evals=100, derivative_system= dumped_pendulum_equation).y[0]
        #theta_actual = np.sign(theta_actual) * (abs(theta_actual) % (np.pi))

        theta_nn = []
        for t in t_range:
            theta_nn.append(my_k.predict(t, theta_0, theta_dot_0, device)[0])
        row = i // 2
        col = i % 2
        
        axs[row, col].plot(t_range, theta_actual, label='Actual')
        axs[row, col].plot(t_range, theta_nn, label='Neural Network')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Theta')
        axs[row, col].set_title('$\\theta_0$=%s, $\dot{\\theta}_0$=%s' % (theta_0, theta_dot_0))
        axs[row, col].legend()
        # draw horizontal line where inference starts
        axs[row, col].axvline(x=2, color='red', linestyle='--')
    plt.suptitle("Trained up to t=2")
    plt.tight_layout()
    plt.savefig('theta_vs_time_Koopman.png')
    plt.show()

    ## Analyze latent space
    my_k = Koopman_autoencoder(2, 64, 2)
    my_k = torch.compile(my_k)
    my_k.load_state_dict(torch.load('K_autoencoder.pth'))
    my_k.to(device)
    my_k.eval()
    # Make a grid of speed and angle, check each of the latent dimensions in these
    theta_range = np.linspace(-np.pi, np.pi, 100)
    theta_dot_range = np.linspace(-np.pi, np.pi, 100)
    latent_space = []
    for theta in theta_range:
        for theta_dot in theta_dot_range:
            latent_space.append([theta, theta_dot])
    latent_space = torch.tensor(latent_space, dtype=torch.float32)
    
    latent_space = latent_space.to(device)
    z_next = my_k.encoder(latent_space)
    z_next = z_next.cpu().detach().numpy()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(z_next[:, 0].reshape(100, 100), cmap='hot', extent=[-np.pi, np.pi, -np.pi, np.pi])
    plt.colorbar()
    plt.xlabel('Theta')
    plt.ylabel('Theta_dot')
    plt.title('Latent Dimension 1')
    plt.subplot(1, 2, 2)
    plt.imshow(z_next[:, 1].reshape(100, 100), cmap='hot', extent=[-np.pi, np.pi, -np.pi, np.pi])
    plt.colorbar()
    plt.xlabel('Theta')
    plt.ylabel('Theta_dot')
    plt.title('Latent Dimension 2')
    plt.tight_layout()
    plt.savefig('latent_space_heatmaps.png')
    plt.show()

    # Now let's visualize the magnitude and phase of the latent space

    magnitude = np.sqrt(z_next[:, 0]**2 + z_next[:, 1]**2)
    phase = np.arctan2(z_next[:, 1], z_next[:, 0])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(magnitude.reshape(100, 100), cmap='hot', extent=[-np.pi, np.pi, -np.pi, np.pi])
    plt.colorbar()
    plt.xlabel('Theta')
    plt.ylabel('Theta_dot')
    plt.title('Magnitude')
    plt.subplot(1, 2, 2)
    plt.imshow(phase.reshape(100, 100), cmap='hot', extent=[-np.pi, np.pi, -np.pi, np.pi])
    plt.colorbar()
    plt.xlabel('Theta')
    plt.ylabel('Theta_dot')
    plt.title('Phase')
    plt.tight_layout()
    plt.savefig('magnitude_phase_heatmaps.png')
    plt.show()

    # Let's visualize one trajectory in the latent space
   
    sol = solve_pendulum(z0 = [np.pi/4, 0], t_range=(0, 10), evals=100, derivative_system=dumped_pendulum_equation)
    latent_trajectory = []
    for i in range(1, len(sol.t)):
        latent_trajectory.append(my_k.encoder(torch.tensor(sol.y[:, i], dtype=torch.float32).unsqueeze(0).to(device)).cpu().detach().numpy()[0])
    latent_trajectory = np.array(latent_trajectory)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # Plot the latent space trajectory
    axs[0].plot(latent_trajectory[:, 0], latent_trajectory[:, 1], label='Latent Space', color='blue')
    # add the time at each point
    for i, txt in enumerate(sol.t[1:]):
        if i % 10 == 0:
            axs[0].annotate(f't = {np.round(txt, 2)}', (latent_trajectory[i, 0], latent_trajectory[i, 1]))
    axs[0].set_xlabel('Latent Dimension 1')
    axs[0].set_ylabel('Latent Dimension 2')
    axs[0].set_title('Latent Space Trajectory')
    axs[0].legend()
    axs[0].set_aspect('equal', adjustable='box')  # Set equal aspect ratio
    

    # Set the same limits for x and y to ensure the plot is square
    max_limit = max(max(latent_trajectory[:, 0]), max(latent_trajectory[:, 1]))
    min_limit = min(min(latent_trajectory[:, 0]), min(latent_trajectory[:, 1]))
    axs[0].set_xlim(min_limit, max_limit)
    axs[0].set_ylim(min_limit, max_limit)
    
    # Plot the phase space trajectory
    axs[1].plot(sol.y[0, :], sol.y[1, :], label='Actual', color='red')
    # add the time at each point
    for i, txt in enumerate(sol.t[1:]):
        if i % 10 == 0:
            axs[1].annotate(f't = {np.round(txt, 2)}', (sol.y[0, i], sol.y[1, i]))
    axs[1].set_xlabel(r'$\theta$')
    axs[1].set_ylabel(r'$\dot{\theta}$')
    axs[1].set_title('Phase Space Trajectory')
    axs[1].legend()
    axs[1].set_aspect('equal', adjustable='box')  # Set equal aspect ratio

    # Set the same limits for x and y to ensure the plot is square
    max_limit = max(max(sol.y[0, :]), max(sol.y[1, :]))
    min_limit = min(min(sol.y[0, :]), min(sol.y[1, :]))
    axs[1].set_xlim(min_limit, max_limit)
    axs[1].set_ylim(min_limit, max_limit)

    # Add grid to the plot
    axs[0].grid()
    axs[1].grid()
    # Add subtitle with trajectory information
    plt.suptitle(r"Latent and Phase Space Trajectory for $\theta_0= 0.25\pi$, $\dot{{\theta}}_0=0$")
    plt.tight_layout()
    plt.savefig('latent_and_phase_space_trajectory.png')
    plt.show()
