import numpy as np
import scipy.io as sio

from hazard_curves import StormSim_JPM

# Loading MATLAB input data
data = sio.loadmat("./data/jpm_input.mat")["data"]


# Configuring and executing JPM code
response_data = dict(
    data=data,
    flag_values=[],
    Ua=0.3738,
    Ur=0.5840,
    tide_std=0.1,
)

jpm_options = dict(
    integration_method=2,
    uncertainty_treatment="combined",
    tide_application=0,
    use_AEP=0,
    ind_Skew=0,
    prc=[16, 84],
)

results = StormSim_JPM(response_data, jpm_options)


# Loading MATLAB out data for comparison
keys = ["HC_plt", "HC_tbl", "HC_tbl_rsp_x", "HC_tbl_rsp_y", "HC_plt_x", "HC_tbl_x"]
out_data = sio.loadmat("./data/jpm_output.mat")["data"]
out_data = {k: out_data[0][0][i] for i, k in enumerate(keys, start=1)}

# Comparing Python results with MATLAB
for k in out_data.keys():
    a = results[k]
    b = out_data[k]
    # Converting 1D array to 2D Nx1 array
    if a.ndim == 1:
        a = a[:, None]

    err = np.abs(a - b)
    err_max = np.nanmax(err)
    print(f"{k}: {err_max}")
