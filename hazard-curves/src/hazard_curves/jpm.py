from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import norm

from .common import aef2aep, bool_check, dict_to_dataclass, get_grid_values


@dataclass
class ResponseData:
    data: np.ndarray  # N x 4 [time, response, skew_tides, dsw]
    flag_value: Optional[float] = None
    Ua: float = 0.0
    Ur: float = 0.0
    SLC: float = 0.0
    tide_std: float = 0.0


@dataclass
class JPMOptions:
    integration_method: int  # 1 = ATCS, 2 = ITCS
    uncertainty_treatment: str  # 'absolute' | 'relative' | 'combined'
    tide_application: int  # 0, 1, 2
    ind_Skew: int  # 0 or 1
    use_AEP: int  # 0 or 1
    prc: List[float]
    stat_print: int = 0


def StormSim_JPM(response_data_dict: dict, jpm_options_dict: dict):

    response_data = dict_to_dataclass(ResponseData, response_data_dict)
    jpm_options = dict_to_dataclass(JPMOptions, jpm_options_dict)

    response_data, jpm_options = error_handeling(
        response_data,
        jpm_options,
    )

    data = response_data.data

    if response_data.flag_value is not None:
        data = data[data[:, 1] != response_data.flag_value]

    mask = np.isfinite(data[:, 1]) & (data[:, 1] > 0)
    data = data[mask]

    if data.size == 0:
        raise RuntimeError("Error: Invalid response removal yielded empty dataset.")

    response_data.data = data

    JPM_output = StormSim_JPM_integration(response_data, jpm_options)

    return JPM_output


def error_handeling(response_data: ResponseData, jpm_options: JPMOptions):

    if (
        not isinstance(response_data.data, np.ndarray)
        or response_data.data.shape[1] != 4
    ):
        raise ValueError(
            "response_data.data must be N x 4 [time, response, skew_tides, dsw]"
        )

    if response_data.SLC < 0 or not np.isfinite(response_data.SLC):
        raise ValueError("response_data.SLC must be a positive scalar")

    checks = [
        (jpm_options, "use_AEP", [0, 1], "0 (AEF) or 1 (AEP)s"),
        (jpm_options, "ind_Skew", [0, 1], "0 (dont apply) or 1 (apply) skew tides"),
        (jpm_options ,"integration_method", [1, 2], "1 (PCHA ATCS) or 2 (PCHA ITCS)"),
    ]
    bool_check(checks)


    #for value, allowed, name in bool_fields:
     #   bool_check(value, allowed, name)

    # --- Tide application ---
    if jpm_options.tide_application == 0:
        response_data.tide_std = 0
        response_data.data[:, 2] = 0
        jpm_options.ind_Skew = 0

    elif jpm_options.tide_application == 1:
        if response_data.tide_std <= 0:
            raise ValueError("tide_std must be positive")
        response_data.data[:, 2] = 0
        jpm_options.ind_Skew = 0

    elif jpm_options.tide_application == 2:
        if jpm_options.ind_Skew == 1:
            if not np.all(np.isfinite(response_data.data[:, 2])):
                raise ValueError("Skew tides must be finite")
            response_data.tide_std = 0
        else:
            if response_data.tide_std <= 0:
                raise ValueError("tide_std must be positive")
            response_data.data[:, 2] = 0

    # --- Uncertainty treatment ---
    if jpm_options.uncertainty_treatment == "relative":
        response_data.Ua = 0.0
    elif jpm_options.uncertainty_treatment == "absolute":
        response_data.Ur = 0.0
    elif jpm_options.uncertainty_treatment != "combined":
        raise ValueError("Invalid uncertainty_treatment")

    if response_data.Ua < 0 or response_data.Ur < 0:
        raise ValueError("Uncertainties must be non-negative")

    # --- Percentiles ---
    if not jpm_options.prc:
        jpm_options.prc = [2.28, 15.87, 84.13, 97.72]
    if len(jpm_options.prc) > 4 or any(p < 0 for p in jpm_options.prc):
        raise ValueError("prc must contain 1–4 values in [0, 100]")

    jpm_options.prc = sorted(jpm_options.prc)

    return response_data, jpm_options


def StormSim_JPM_integration(response_data, jpm_options):

    # --------------------------------------------------
    # INITIALIZE RESPONSE FIELDS
    # --------------------------------------------------
    Resp = response_data.data[:, 1].astype(float)  # Response
    ProbMass = response_data.data[:, 3].astype(float)  # DSW
    U_tide = response_data.data[:, 2].astype(float)

    U_a = response_data.Ua
    U_r = response_data.Ur

    randomNorm = 0.0

    # Remove implicit SLC
    Resp = Resp - response_data.SLC

    # --------------------------------------------------
    # INITIALIZE HAZARD FREQUENCIES
    # --------------------------------------------------
    HC_tbl_x, HC_tbl_rsp_y, HC_plt_x = get_grid_values(jpm_options.use_AEP == 1)

    # --------------------------------------------------
    # PCHA ITCS ONLY
    # --------------------------------------------------
    if jpm_options.integration_method == 2:
        # Discrete Gaussian Z-scores
        dscrtGauss = get_gaussian_norm()
        nReps = len(dscrtGauss)

        dscrt = np.sort(np.tile(dscrtGauss, len(Resp)))

        ProbMass = np.tile(ProbMass / nReps, nReps)
        Resp = np.tile(Resp, nReps)
        U_tide = np.tile(U_tide, nReps)

        # Partition uncertainties
        p1_a = 0.1
        if U_a**2 >= p1_a**2:
            U_a = np.sqrt(U_a**2 - p1_a**2)

        p1_r = 0.1
        if U_r**2 >= p1_r**2:
            U_r = np.sqrt(U_r**2 - p1_r**2)

        # Apply first partition
        Resp = Resp + dscrt * (p1_a + Resp * p1_r) / 2.0

    # --------------------------------------------------
    # TIDES
    # --------------------------------------------------
    resp_sz = len(Resp)

    match jpm_options.tide_application:
        case 0:
            tide_cl = 0.0

        case 1:
            tide_cl = response_data.tide_std

        case 2:
            tide_cl = 0.0

            if jpm_options.ind_Skew == 0:
                np.random.seed(0)

                if jpm_options.integration_method == 1:
                    randomNorm = np.random.randn(resp_sz)
                else:
                    randomNorm = dscrt

                U_tide = response_data.tide_std
        case _:
            randomNorm = 1.0

    # Add tides + SLC back
    Resp = Resp + randomNorm * U_tide + response_data.SLC

    # --------------------------------------------------
    # CONFIDENCE LIMITS
    # --------------------------------------------------
    z = norm.ppf(np.array(jpm_options.prc) / 100.0)

    match jpm_options.uncertainty_treatment:

        case "absolute":

            def CL_unc(y, U_t):
                return y[:, None] + z[None,:] * np.sqrt(U_a**2 + U_t**2)

        case "relative":

            def CL_unc(y, U_t):
                return y[:, None] + z[None,:] * y[:, None] * np.sqrt((y[:, None] * U_r) ** 2 + U_t**2)

        case "combined":

            if tide_cl == 0:

                def CL_unc(y, U_t):
                    return y[:, None] + z[None,:] / np.sqrt(1 / U_a**2 + 1 / (y[:,None] * U_r) ** 2)

            else:

                def CL_unc(y, U_t):
                    return y[:, None] + z[None] / np.sqrt(
                        1 / U_a**2 + 1 / (y[:, None] * U_r) ** 2 + 1 / U_t**2
                    )

    # --------------------------------------------------
    # PERFORM INTEGRATION
    # --------------------------------------------------
    mask = np.isfinite(Resp) & (Resp > 0)
    Resp = Resp[mask]
    ProbMass = ProbMass[mask]


    idx = np.argsort(Resp)[::-1]
    y = Resp[idx]
    x = np.cumsum(ProbMass[idx])

    x[x == 0] = 1e-16



    # Remove duplicates (x then y)
    _, ia = np.unique(x, return_index=True)
    x, y = x[ia], y[ia]

    _, iy = np.unique(y, return_index=True)
    x, y = x[iy], y[iy]

    # Sort ascending in x
    order = np.argsort(x)
    x = np.log(x[order])
    y = y[order]
    # Percentiles
    resp_perc = CL_unc(y, tide_cl)
    y = np.column_stack([y, resp_perc])

    # --------------------------------------------------
    # INTERPOLATION
    # --------------------------------------------------
    #
    #print(HC_plt_x.shape)
    #print(np.log(HC_plt_x), x, y)

    

    n = y.shape[1]

    y_plt = np.full((len(HC_plt_x), n), np.nan)


    HC_tbl_y = np.full((len(HC_tbl_x), n), np.nan)
    HC_tbl_rsp_x = np.full((len(HC_tbl_rsp_y), n), np.nan)

    for i in range(n):
        y_plt[:,i] = np.interp(np.log(HC_plt_x), x, y[:,i], left=np.nan, right=np.nan)
        HC_tbl_y[:,i] = np.interp(np.log(HC_tbl_x), x, y[:,i], left=np.nan, right=np.nan)
    
        yi = y[:, i]
        _, iu = np.unique(yi, return_index=True)
        yi_u = yi[iu]
        xi_u = x[iu]

        HC_tbl_rsp_x[:, i] = np.exp(
            np.interp(HC_tbl_rsp_y, yi_u, xi_u, left=np.nan, right=np.nan)
        )

    # Remove negatives
    HC_tbl_y[HC_tbl_y < 0] = np.nan
    HC_tbl_rsp_x[HC_tbl_rsp_x < 0] = np.nan
    y_plt[y_plt < 0] = np.nan

    # Convert to AEP if required
    if jpm_options.use_AEP == 1:
        HC_tbl_rsp_x = aef2aep(HC_tbl_rsp_x)

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    return {
        "HC_plt": y_plt,
        "HC_tbl": HC_tbl_y,
        "HC_tbl_rsp_x": HC_tbl_rsp_x,
        "HC_tbl_rsp_y": HC_tbl_rsp_y,
        "HC_plt_x": HC_plt_x,
        "HC_tbl_x": HC_tbl_x,
    }

#fmt: off
def get_gaussian_norm():
    return np.array([
        -3.0004, -2.6927, -2.525, -2.4072, -2.3156, -2.2401, -2.1755, -2.1189,
        -2.0683, -2.0226, -1.9807, -1.942, -1.9061, -1.8725, -1.8408, -1.8109,
        -1.7825, -1.7555, -1.7297, -1.7051, -1.6814, -1.6586, -1.6367, -1.6155,
        -1.595, -1.5752, -1.556, -1.5373, -1.5192, -1.5015, -1.4843, -1.4675,
        -1.4511, -1.4352, -1.4195, -1.4042, -1.3893, -1.3746, -1.3602, -1.3461,
        -1.3323, -1.3187, -1.3054, -1.2922, -1.2793, -1.2666, -1.2542, -1.2419,
        -1.2297, -1.2178, -1.206, -1.1944, -1.183, -1.1717, -1.1606, -1.1496,
        -1.1387, -1.128, -1.1174, -1.1069, -1.0966, -1.0863, -1.0762, -1.0662,
        -1.0563, -1.0465, -1.0367, -1.0271, -1.0176, -1.0082, -0.9988, -0.9896,
        -0.9804, -0.9713, -0.9623, -0.9534, -0.9445, -0.9358, -0.927, -0.9184,
        -0.9098, -0.9013, -0.8929, -0.8845, -0.8762, -0.8679, -0.8598, -0.8516,
        -0.8435, -0.8355, -0.8275, -0.8196, -0.8117, -0.8039, -0.7961, -0.7884,
        -0.7807, -0.7731, -0.7655, -0.758, -0.7505, -0.743, -0.7356, -0.7282,
        -0.7209, -0.7136, -0.7063, -0.6991, -0.6919, -0.6848, -0.6776, -0.6706,
        -0.6635, -0.6565, -0.6495, -0.6426, -0.6356, -0.6288, -0.6219, -0.6151,
        -0.6083, -0.6015, -0.5947, -0.588, -0.5813, -0.5746, -0.568, -0.5614,
        -0.5548, -0.5482, -0.5417, -0.5351, -0.5286, -0.5221, -0.5157, -0.5093,
        -0.5028, -0.4964, -0.4901, -0.4837, -0.4774, -0.4711, -0.4648, -0.4585,
        -0.4523, -0.446, -0.4398, -0.4336, -0.4274, -0.4212, -0.4151, -0.4089,
        -0.4028, -0.3967, -0.3906, -0.3845, -0.3784, -0.3724, -0.3663, -0.3603,
        -0.3543, -0.3483, -0.3423, -0.3363, -0.3304, -0.3244, -0.3185, -0.3125,
        -0.3066, -0.3007, -0.2948, -0.2889, -0.283, -0.2772, -0.2713, -0.2655,
        -0.2596, -0.2538, -0.248, -0.2422, -0.2363, -0.2306, -0.2248, -0.219,
        -0.2132, -0.2074, -0.2017, -0.1959, -0.1902, -0.1844, -0.1787, -0.173,
        -0.1672, -0.1615, -0.1558, -0.1501, -0.1444, -0.1387, -0.133, -0.1273,
        -0.1216, -0.1159, -0.1103, -0.1046, -0.0989, -0.0932, -0.0876, -0.0819,
        -0.0763, -0.0706, -0.0649, -0.0593, -0.0536, -0.048, -0.0423, -0.0367,
        -0.031, -0.0254, -0.0197, -0.0141, -0.0085, -0.0028, 0.0028, 0.0085,
        0.0141, 0.0197, 0.0254, 0.031, 0.0367, 0.0423, 0.048, 0.0536, 0.0593,
        0.0649, 0.0706, 0.0763, 0.0819, 0.0876, 0.0932, 0.0989, 0.1046, 0.1103,
        0.1159, 0.1216, 0.1273, 0.133, 0.1387, 0.1444, 0.1501, 0.1558, 0.1615,
        0.1672, 0.173, 0.1787, 0.1844, 0.1902, 0.1959, 0.2017, 0.2074, 0.2132,
        0.219, 0.2248, 0.2306, 0.2363, 0.2422, 0.248, 0.2538, 0.2596, 0.2655,
        0.2713, 0.2772, 0.283, 0.2889, 0.2948, 0.3007, 0.3066, 0.3125, 0.3185,
        0.3244, 0.3304, 0.3363, 0.3423, 0.3483, 0.3543, 0.3603, 0.3663, 0.3724,
        0.3784, 0.3845, 0.3906, 0.3967, 0.4028, 0.4089, 0.4151, 0.4212, 0.4274,
        0.4336, 0.4398, 0.446, 0.4523, 0.4585, 0.4648, 0.4711, 0.4774, 0.4837,
        0.4901, 0.4964, 0.5028, 0.5093, 0.5157, 0.5221, 0.5286, 0.5351, 0.5417,
        0.5482, 0.5548, 0.5614, 0.568, 0.5746, 0.5813, 0.588, 0.5947, 0.6015,
        0.6083, 0.6151, 0.6219, 0.6288, 0.6356, 0.6426, 0.6495, 0.6565, 0.6635,
        0.6706, 0.6776, 0.6848, 0.6919, 0.6991, 0.7063, 0.7136, 0.7209, 0.7282,
        0.7356, 0.743, 0.7505, 0.758, 0.7655, 0.7731, 0.7807, 0.7884, 0.7961,
        0.8039, 0.8117, 0.8196, 0.8275, 0.8355, 0.8435, 0.8516, 0.8598, 0.8679,
        0.8762, 0.8845, 0.8929, 0.9013, 0.9098, 0.9184, 0.927, 0.9358, 0.9445,
        0.9534, 0.9623, 0.9713, 0.9804, 0.9896, 0.9988, 1.0082, 1.0176, 1.0271,
        1.0367, 1.0465, 1.0563, 1.0662, 1.0762, 1.0863, 1.0966, 1.1069, 1.1174,
        1.128, 1.1387, 1.1496, 1.1606, 1.1717, 1.183, 1.1944, 1.206, 1.2178,
        1.2297, 1.2419, 1.2542, 1.2666, 1.2793, 1.2922, 1.3054, 1.3187, 1.3323,
        1.3461, 1.3602, 1.3746, 1.3893, 1.4042, 1.4195, 1.4352, 1.4511, 1.4675,
        1.4843, 1.5015, 1.5192, 1.5373, 1.556, 1.5752, 1.595, 1.6155, 1.6367,
        1.6586, 1.6814, 1.7051, 1.7297, 1.7555, 1.7825, 1.8109, 1.8408, 1.8725,
        1.9061, 1.942, 1.9807, 2.0226, 2.0683, 2.1189, 2.1755, 2.2401, 2.3156,
        2.4072, 2.525, 2.6927, 3.0004
    ])
#fmt:on 
