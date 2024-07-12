import sys
sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn
from pendulum import solve_pendulum, pendulum_equation, dumped_pendulum_equation
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class snake(nn.Module):
    def forward(self, x):
        return x + torch.sin(x)**2



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 32), 
            snake(),
            nn.Linear(32, 64),
            snake(),
            nn.Linear(64, 128),
            snake(),
            nn.Linear(128, 64),
            snake(),
            nn.Linear(64, 32),
            snake(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        angle_in_radians = torch.pi * torch.tanh(output)       
        return angle_in_radians

def generate_data(n_samples, damped = False, t_range=(0, 10), evals=1000):
    """
    The idea is to generate tuples of (t, theta_0, theta_dot_0, theta_t). 
    """

    initial_theta_range = [-np.pi, np.pi]
    #initial_theta_dot_range = [-np.pi, np.pi] # This is change of position per unit of time. 
                                             # \pi velocity means that the pendulum goes from one extreme to the other in one unit of time.
    
    data = []
    for _ in range(n_samples):
        z0 = [np.random.uniform(*initial_theta_range), 0]
        if damped:
            sol = solve_pendulum(z0=z0, t_range=t_range, derivative_system=dumped_pendulum_equation, evals = evals)
        else:
            sol = solve_pendulum(z0=z0, t_range=t_range, evals = evals)
        for i in range(1, len(sol.t)):
            data.append((sol.t[i], z0[0], z0[1], np.sign(sol.y[0, i]) * (abs(sol.y[0, i]) % (np.pi)), sol.y[1, i]))
    
    # suffle the data
    np.random.shuffle(data)
    # assert that \theta is between -\pi and \pi
    assert all(-np.pi <= d[3] <= np.pi for d in data)
    return data

class train_network:
    def __init__(self, epochs, n_samples, damped = False, lr=1e-3, batch_size=64, second_derivative_penalty=False):
        self.epochs = epochs
        self.second_derivative_penalty = second_derivative_penalty
        self.n_samples = n_samples
        self.damped = damped
        self.model = NeuralNetwork()
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=lr)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.loss_fn = nn.MSELoss()
        self.second_derivative_loss = nn.MSELoss()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def train(self):
        self.data_loader_train = self.generate_data(self.n_samples, batch_size= self.batch_size, t_range=(0, 2), evals= 200)
        self.data_loader_val = self.generate_data(self.n_samples // 6, batch_size= self.batch_size * 3, t_range=(2, 10), evals= 200)
        self.loss_dict = {}
        self.loss_dict['train'] = []
        self.loss_dict['val'] = []
        early_stopping = self.early_stopping(patience=100)
        for epoch in range(self.epochs):
            loss_list = []
            #pbar = tqdm(self.data_loader_train)
            for X, y in self.data_loader_train:
                #y = y.unsqueeze(1)
                self.optimizer.zero_grad()
                X.requires_grad = True
                y_pred = self.model(X)
                if self.second_derivative_penalty:
                    loss = self.get_second_derivative_loss(X, y_pred) + self.loss_fn(y_pred, y)
                else:
                    loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
                #pbar.set_description(f"Training Loss: {np.mean(loss_list)}")
            #self.scheduler.step()
            if epoch % 5 == 0:
                print(f"Epoch: {epoch}, Loss: {np.mean(loss_list)}")
            self.loss_dict['train'].append(np.mean(loss_list))
            #with torch.no_grad():
            loss_list = []
            #pbar = tqdm(self.data_loader_val)
            for X, y in self.data_loader_val:
                #y = y.unsqueeze(1)
                X.requires_grad = True
                y_pred = self.model(X)
                if self.second_derivative_penalty:
                    loss = self.get_second_derivative_loss(X, y_pred) + self.loss_fn(y_pred, y)
                else:
                    loss = self.loss_fn(y_pred, y)
                loss_list.append(loss.item())
                #pbar.set_description(f"Validation Loss MSE: {np.mean(loss_list)}")
            self.loss_dict['val'].append(np.mean(loss_list))
            
            #if early_stopping(self.loss_dict['val'][-1]):
            #    print(f"Early stopping at epoch {epoch}")
            #    break
            if epoch % 5 == 0:
                if self.second_derivative_penalty:
                    self.save_model(f'pendulum_model_pinn.pth')
                else:
                    self.save_model(f'pendulum_model_vanilla.pth')

                print(f"Epoch: {epoch}, Val Loss: {np.mean(loss_list)}")

    def predict(self, t, theta_0, theta_dot_0):
        t = torch.tensor(t, dtype=torch.float32)
        theta_0 = torch.tensor(theta_0, dtype=torch.float32)
        theta_dot_0 = torch.tensor(theta_dot_0, dtype=torch.float32)
        input = torch.tensor([t, theta_0, theta_dot_0], dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.model(input).cpu().detach().numpy()[0]
    
    def generate_data(self, n_samples, batch_size=64, t_range=(0, 10), evals=1000):
        data_list = generate_data(n_samples, self.damped, t_range=t_range, evals=evals)
        data = np.array(data_list)
        X = data[:, :-2]
        y = data[:, -2:-1]
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).to(self.device), torch.tensor(y, dtype=torch.float32).to(self.device))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def plot_loss(self):
        plt.plot(self.loss_dict['train'], label='Train')
        plt.plot(self.loss_dict['val'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if self.second_derivative_penalty:
            plt.savefig('loss_pinn.png')
        else:
            plt.savefig('loss.png')
        plt.show()

    class early_stopping:
        def __init__(self, patience):
            self.patience = patience
            self.counter = 0
            self.best_loss = np.inf
        def __call__(self, loss):
            if loss < self.best_loss:
                self.best_loss = loss
                self.counter = 0
            else:
                self.counter += 1
            return self.counter > self.patience
    
    def get_second_derivative_loss(self, X, y_pred):
        # 1st derivative:
        # \dot{\theta} = \theta_dot
        X.requires_grad = True
        dtheta = torch.autograd.grad(y_pred[:, 0], X, grad_outputs=torch.ones_like(y_pred[:, 0]), retain_graph= True, create_graph=True)[0]
        #print(dtheta)
        # 2nd derivative:
        # \ddot{\theta} = \dot{\theta_dot}
        predicted_ddtheta = torch.autograd.grad(dtheta, X, grad_outputs=torch.ones_like(dtheta), retain_graph= True, create_graph=True)[0][:, 0]
        real_ddtheta = self.real_ddtheta(X)
        #print(f"Predicted: {predicted_ddtheta}")
        #print(f"Real: {real_ddtheta}")
        #exit()
        return self.second_derivative_loss(predicted_ddtheta, real_ddtheta)

    def real_ddtheta(self, X):
        g = 9.8
        c = 0.1
        if self.damped:
            return -g * torch.sin(X[:, 1]) - c * X[:, 2]
        else:
            return -g * torch.sin(X[:, 1])

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

def run_pinn():
    trainer = train_network(epochs=250, n_samples=5000, damped=True, lr=0.0003, batch_size=512, second_derivative_penalty=True)
    trainer.train()
    t_range = np.linspace(0, 10, 100)
    initial_conditions = [
        (0, 0.5*np.pi, 0),
        (0, 0.25*np.pi, 0),
        (0, 0.75*np.pi, 0),
        (0, 0.9*np.pi, 0)
    ]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, (_, theta_0, theta_dot_0) in enumerate(initial_conditions):
        theta_actual = solve_pendulum([theta_0, theta_dot_0], t_range=(0, 10), evals=100, derivative_system=dumped_pendulum_equation).y[0]
        theta_nn = []
        for t in t_range:
            theta_nn.append(trainer.predict(t, theta_0, theta_dot_0))
        row = i // 2
        col = i % 2
        
        axs[row, col].plot(t_range, theta_actual, label='Actual')
        axs[row, col].plot(t_range, theta_nn, label='Neural Network')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Theta')
        axs[row, col].set_title('$\\theta_0$=%s, $\dot{\\theta}_0$=%s' % (theta_0, theta_dot_0))
        axs[row, col].legend()
    plt.suptitle("PINN, trained up to t=10")
    plt.tight_layout()
    plt.savefig('theta_vs_time_PINN.png')
    plt.show()

def run_vanilla_nn():   
    trainer = train_network(epochs=250, n_samples=2000, damped=True, lr=0.01, batch_size=512)
    trainer.model.load_state_dict(torch.load('pendulum_model_vanilla.pth'))
    #trainer.train()
    ## Generate data for plotting
    t_range = np.linspace(0, 10, 100)
    initial_conditions = [
        (0, 0.5*np.pi, 0),
        (0, 0.25*np.pi, 0),
        (0, 0.75*np.pi, 0),
        (0, 0.5*np.pi, 0)
    ]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, (_, theta_0, theta_dot_0) in enumerate(initial_conditions):
        theta_actual = solve_pendulum([theta_0, theta_dot_0], t_range=(0, 10), evals=100, derivative_system=dumped_pendulum_equation).y[0]
        theta_nn = []
        for t in t_range:
            theta_nn.append(trainer.predict(t, theta_0, theta_dot_0))
        row = i // 2
        col = i % 2
        
        axs[row, col].plot(t_range, theta_actual, label='Actual')
        axs[row, col].plot(t_range, theta_nn, label='Neural Network')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Theta')
        axs[row, col].set_title('$\\theta_0$=%s, $\dot{\\theta}_0$=%s' % (theta_0, theta_dot_0))
        axs[row, col].legend()
    plt.suptitle("Vanilla NN, trained up to t=2")
    plt.tight_layout()
    plt.savefig('theta_vs_time_NN.png')
    plt.show()

if __name__ == "__main__":
    run_pinn()
    run_vanilla_nn()