
import torch 
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from IPython.display import set_matplotlib_formats
import matplotlib
import numpy as np
#%% matplotlib setup
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "14"
mps_device = torch.device("mps")


#%% Functions parameters
L = 10                                      # Space domain
Alpha_vect = torch.linspace(0,1,100)        # vector of alphas
x_vect = torch.linspace(0,L,2000)           # vector of x


Function = 'Heaviside'                      # Alpha-parameterised step function
Function = 'Tanh'                           # smooth alpha-parameterised step function
# Function = 'Gauss'                        # Alpha-parameterised front function
# Function = 'Gauss_sum'                      # Double alpha-parameterised front functions




if Function == 'Heaviside':
    F = torch.heaviside((x_vect[:,None] - (1-Alpha_vect[None,:])*L), x_vect[-1]/x_vect[-1])
elif Function == 'Tanh':
    F = torch.tanh((x_vect[:,None] - (1-Alpha_vect[None,:])*L))
elif Function == 'Gauss':
    F = torch.exp(-(x_vect[:,None] - (1-Alpha_vect[None,:])*L)**2)

elif Function == 'Gauss_sum':
    F = torch.exp(-(x_vect[:,None] - (1-Alpha_vect[None,:])*L)**2) + torch.exp(-(x_vect[:,None] - (1-2*Alpha_vect[None,:])*L)**2) 

 
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.N_0 = 2000
        self.N_1 = 1000
        self.N_2 = 200
        self.N_3 = 1
        self.encoder = nn.Sequential(
        nn.Linear(self.N_0,self.N_1),
        nn.Tanh(),
        nn.Linear(self.N_1, self.N_2),
        nn.Tanh(),
        nn.Linear(self.N_2, self.N_3)
        )

        self.decoder = nn.Sequential(
        nn.Linear(self.N_3,self.N_2),
        nn.ReLU(),
        nn.Linear(self.N_2, self.N_1),
        nn.ReLU(),
        nn.Linear(self.N_1, self.N_0)
        )
    
    def forward(self, x, role = "decode"):
        if self.training:
            e = self.encoder(x)
            d = self.decoder(e)
            return d
        else:
            match role:
                case "encode":
                    e = self.encoder(x)
                    return e
                case "decode":
                    d = self.decoder(x)
                    return d

ROM = AutoEncoder()
MSE = nn.MSELoss()

optimizer = torch.optim.Adam(ROM.parameters(),
                             lr = 1e-3)

n_epochs = 400
import random
val = int(np.floor(0.2*F.shape[1]))
F_train = F
for n in range(val):
    r = random.randint(1, F_train.shape[1]-1)
    F_val = F_train[:,r]
    F_train = torch.cat([F_train[:, :r], F_train[:, r+1:]], dim=1)

F_train = F_train.T
F_val = F_val.T

loss_t_vect = []
loss_v_vect = []

#%% train 
ROM.train()
F_train = F_train.to(mps_device)
F_val = F_val.to(mps_device)
ROM.to(mps_device)
import time
t0 = time.time()

for epochs in range(n_epochs):
    loss = MSE(ROM(F_train),F_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_t_vect.append(loss.data)
    loss_v_vect.append(MSE(ROM(F_val),F_val).data)
    print(f'epoch {epochs+1} loss = {np.format_float_scientific(loss.item(), precision=4)}')

tf = time.time()

print(f'duration (s) {tf-t0}')
#%% eval
ROM.eval()

F_train = F_train.cpu()
F_val = F_val.cpu()
ROM.cpu()
torch.save(ROM, 'FullModel.pt') # to save a full coarse model



#%% plots
loss_t_vect = [loss_t.cpu() for loss_t in loss_t_vect]
loss_v_vect = [loss_v.cpu() for loss_v in loss_v_vect]


plt.plot(loss_t_vect,label = 'training set')
plt.plot(loss_v_vect,label = 'validation set')
plt.legend(loc="upper right")
plt.show()
plt.semilogy(loss_t_vect,label = 'training set')
plt.semilogy(loss_v_vect,label = 'validation set')
plt.legend(loc="upper right")
plt.xlabel('Epochs')
plt.xlabel('Loss')
# plt.savefig(f'Results/loss_training_'+Function+'.pdf', transparent=True)  
plt.show()

Alpha_latent = ROM(F.t(),"encode")
plt.plot(Alpha_vect.view(-1,1).cpu().data,Alpha_latent.view(-1,1).cpu().data)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\hat{\alpha}$')
plt.show()
F_reconstructed = ROM(Alpha_latent.view(-1,1))
# F_reconstructed = ROM(Alpha_vect.view(-1,1))

plt.imshow(F_reconstructed.cpu().data,cmap='gray')
plt.show()

#%% comparison

plt.imshow(F_reconstructed.cpu().data-F.t(),cmap='gray')
plt.show()
idx1 = int(np.floor(0.66*F.shape[1]))
idx2 = int(np.floor(0.33*F.shape[1]))

plt.plot(x_vect,F[:,idx1],'k',label='Full, alpha = 2/3')
plt.plot(x_vect,F_reconstructed.t()[:,idx1].cpu().data,'--',label='Truncated, alpha = 2/3')
plt.plot(x_vect,F[:,idx2],'k',label='Full, alpha = 1/3')
plt.plot(x_vect,F_reconstructed.t()[:,idx2].cpu().data,'--',label='Truncated, alpha = 1/3')
plt.legend(loc="upper left")
plt.title(f'2 slices of the field')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x,\alpha)$')
# plt.savefig(f'Results/Sliced_TruncatedField_{N}_'+Function+'.pdf', transparent=True)  
plt.show()

# %% Interactive plot of the truncated function

from ipywidgets import interact, widgets
import torch

def interactive_plot(alpha):
    
    plt.plot(x_vect,F[:,-alpha])
    plt.plot(x_vect,F_reconstructed.t()[:,-alpha].data)
    plt.show()

# Create an interactive slider
# slider_E1 = widgets.IntSlider(value=0, min=1, max=50, step=1, description='N modes')
slider_E2 = widgets.IntSlider(value=0, min=1, max=int(F.shape[1]-1), step=1, description='alpha')


# Connect the slider to the interactive plot function
interactive_plot_widget = interact(interactive_plot, alpha=slider_E2)
# %%
