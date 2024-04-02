
#%% Imports libraries
import torch 
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
import matplotlib

#%% Defines figures's font
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "14"


#%% Functions parameters
L = 10                                      # Space domain
Alpha_vect = torch.linspace(0,1,1500)       # vector of alphas
x_vect = torch.linspace(0,L,2000)           # vector of x


Function = 'Heaviside'                      # Alpha-parameterised step function
Function = 'Tanh'                           # smooth alpha-parameterised step function
# Function = 'Gauss'                        # Alpha-parameterised front function
Function = 'Gauss_sum'                      # Double alpha-parameterised front functions




if Function == 'Heaviside':
    F = torch.heaviside((x_vect[:,None] - (1-Alpha_vect[None,:])*L), x_vect[-1]/x_vect[-1])
elif Function == 'Tanh':
    F = torch.tanh((x_vect[:,None] - (1-Alpha_vect[None,:])*L))
elif Function == 'Gauss':
    F = torch.exp(-(x_vect[:,None] - (1-Alpha_vect[None,:])*L)**2)

elif Function == 'Gauss_sum':
    F = torch.exp(-(x_vect[:,None] - (1-Alpha_vect[None,:])*L)**2) + torch.exp(-(x_vect[:,None] - (1-2*Alpha_vect[None,:])*L)**2) 


#%% Plot the reference function

plt.imshow(F.t(),cmap='gray')
plt.title('Full field')
plt.xlabel(r'$x$')
plt.ylabel(r'$\alpha$')
plt.savefig('Results/FullField_'+Function+'.pdf', transparent=True)  
plt.show()

#%% SVD decomposition of the reference function
U, S, V = torch.svd(F)


#%% Truncation
N = 15                                                  # Number of modes kept for the truncation                   

F_truncated = U[:,:N]@torch.diag(S[:N])@V[:,:N].t()     # Truncated function

#%% Plot the truncated function

plt.imshow(F_truncated.t(),cmap='gray')
plt.title(f'Truncated field, N={N}')
plt.xlabel(r'$x$')
plt.ylabel(r'$\alpha$')
plt.savefig(f'Results/TruncatedField_{N}_'+Function+'.pdf', transparent=True)  
plt.show()


#%% Plot a comparison between the truncated and the reference functions

plt.plot(x_vect,F[:,1000],'k',label='Full, alpha = 2/3')
plt.plot(x_vect,F_truncated[:,1000],'--',label='Truncated, alpha = 2/3')
plt.plot(x_vect,F[:,500],'k',label='Full, alpha = 1/3')
plt.plot(x_vect,F_truncated[:,500],'--',label='Truncated, alpha = 1/3')
plt.legend(loc="upper left")
plt.title(f'2 slices of the field, N={N}')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x,\alpha)$')
plt.savefig(f'Results/Sliced_TruncatedField_{N}_'+Function+'.pdf', transparent=True)  
plt.show()

#%% Plot the decay of the singular values

plt.semilogy(S)
plt.ylabel(r'$\sigma_i^2$')
plt.xlabel(r'Modes')
plt.savefig(f'Results/SVD_Decay_'+Function+'.pdf', transparent=True)  
plt.show()

# %% Interactive plot of the truncated function

from ipywidgets import interact, widgets
import torch

def interactive_plot(N,alpha):
    
    # Calculate the F_trunc
    F_truncated = U[:,:N]@torch.diag(S[:N])@V[:,:N].t()

    # Plot the function
    plt.plot(x_vect,F[:,alpha])
    plt.plot(x_vect,F_truncated[:,alpha])
    plt.show()

# Create an interactive slider
slider_E1 = widgets.IntSlider(value=0, min=1, max=50, step=1, description='N modes')
slider_E2 = widgets.IntSlider(value=0, min=1, max=1499, step=1, description='alpha')


# Connect the slider to the interactive plot function
interactive_plot_widget = interact(interactive_plot, N=slider_E1, alpha=slider_E2)

# %%
