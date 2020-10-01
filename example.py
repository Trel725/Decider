import numpy as np
import seaborn as sns
import pandas as pd
from custom_utils import norm_func
import matplotlib.pyplot as plt
from custom_utils import cuda_memgrowth
from tensorflow.keras.models import load_model
from BSFilter2 import BSFilter
import glob
from Decider import Decider
cuda_memgrowth()


#%%
nnmodel = load_model("/home/user/Documents/projects/dna_complement/main_model_final.h5",
                     custom_objects={"BSFilter": BSFilter})
x_train = np.load("/home/user/Documents/projects/dna_complement/X_train.npy"),
y_train = np.load("/home/user/Documents/projects/dna_complement/Y_train.npy")

#%%
decider = Decider(nnmodel, x_train, y_train,
                  epsilon=0.075, prior_compl=0.5, n_samples_prediction=200, n_samples_reference=20)

#decider.update_params(lmbd_fn=3, lmbd_fp=1, epsilon=0.1)

#%%
for file in sorted(glob.glob("/home/user/Documents/projects/dna_complement/unknown/unknown/joined/9_pr.csv")):
    print(file)
    sample = pd.read_csv(file)
    sample = sample.apply(norm_func, axis=1).values[:, :2089]
    decider.decide_samples([sample])
    #decider.visualize_sample()
    #plt.show()
    print("-" * 10 + "\n")

#%%
style1((4.5,3))

decider.visualize_intervals()
plt.xlabel("z")
plt.ylabel("p")

grid = np.arange(decider.startgrid, decider.endgrid, 0.01)
theta_compl_idx = np.argmin(np.abs(decider.theta_compl -
                                           (decider.spline_compl(grid) / decider.spline_noncompl(grid))))
theta_noncompl_idx = np.argmin(np.abs(decider.theta_noncompl -
                                              (decider.spline_noncompl(grid) / decider.spline_compl(grid))))
        
plt.xticks([-2, -1, 0, 1, 2])
plt.axvline(grid[theta_noncompl_idx], linewidth=0.9, c="k")
plt.axvline(grid[theta_compl_idx], linewidth=0.9, c="k")


plt.text(grid[theta_compl_idx]-0.05, -0.1, r"$\theta_{+}$", size=12)
plt.text(grid[theta_noncompl_idx]-0.05, -0.1, r"$\theta_{−}$", size=12)


plt.tight_layout(0.25)

plt.savefig("densities.png", dpi=300)


