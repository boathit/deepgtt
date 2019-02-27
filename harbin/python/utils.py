import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_utils import DataLoader
from model import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(fname):
    ## loading gpu model into cpu
    model = torch.load(fname, map_location=lambda storage, loc: storage)
    params = model["params"]
    probrho = ProbRho(params["num_u"], params["dim_u"], params["dict_u"], params["lengths"],
                      params["num_s1"], params["dim_s1"], params["dict_s1"],
                      params["num_s2"], params["dim_s2"], params["dict_s2"],
                      params["num_s3"], params["dim_s3"], params["dict_s3"],
                      params["hidden_size1"], params["dim_rho"],
                      params["dropout"], params["use_selu"], device).to(device)
    probtraffic = ProbTraffic(1, params["hidden_size2"], params["dim_c"],
                              params["dropout"], params["use_selu"]).to(device)
    probttime = ProbTravelTime(params["dim_rho"], params["dim_c"], params["hidden_size3"],
                               params["dropout"], params["use_selu"]).to(device)
    probrho.load_state_dict(model["probrho"])
    probtraffic.load_state_dict(model["probtraffic"])
    probttime.load_state_dict(model["probttime"])
    return probrho, probtraffic, probttime

def log_pdf(logμ, logλ, t):
    """
    log pdf of IG distribution.
    Input:
      logμ, logλ (batch_size, ): the parameters of IG
      t (batch_size, ): the travel time
    ------
    Output:
      logpdf (batch_size, )
    """
    eps = 1e-9
    μ = torch.exp(logμ)
    expt = -0.5 * torch.exp(logλ)*torch.pow(t-μ,2) / (μ.pow(2)*t+eps)
    logz = 0.5*logλ - 1.5*torch.log(t) - 0.9189
    return expt+logz

def trip2logIG(trips, ratios, S):
    """
    data = dataloader.random_emit(10)
    trip2logIG(probrho, data.trips, data.ratios, data.S)
    """
    road_lens = probrho.roads_length(trips, ratios)
    l = road_lens.sum(dim=1) # the whole trip lengths
    w = road_lens / road_lens.sum(dim=1, keepdim=True) # road weights
    rho = probrho(trips)
    c, mu_c, logvar_c = probtraffic(S.to(device))
    logμ, logλ = probttime(rho, c, w, l)
    return logμ, logλ

def expectation(trips, ratios, S):
    """
    Return the expected travel time of the trips.
    """
    logμ, logλ = trip2logIG(trips, ratios, S)
    return torch.exp(logμ)

def hour_dist(hour, trips, S, t):
    """
    Return the probability density values of a road (specified by `trips`) at a given
    `hour`.
    """
    ratios = torch.ones(trips.shape, dtype=torch.float32)
    logμ, logλ = trip2logIG(trips,
                            ratios, S[hour].unsqueeze(0).unsqueeze(0))
    logpdf = log_pdf(logμ.view(1, 1),
                     logλ.view(1, 1),
                     t.to(device))
    y = torch.exp(logpdf.detach()).cpu().numpy().ravel()
    return y

def predict(dataloader, batch_size=128):
    slot_size = np.array(list(map(lambda s:s.ntrips, dataloader.slotdata_pool)))
    print("There are {} trips in total".format(slot_size.sum()))
    num_iterations = int(np.ceil(slot_size/batch_size).sum())
    ŷ, y, d = [], [], []
    for i in range(num_iterations):
        if i % 500 == 0: print("Passed {} batches.".format(i))
        data = dataloader.order_emit(batch_size)
        μ = expectation(data.trips, data.ratios, data.S)
        ŷ.append(μ.detach())
        y.append(data.times)
        d.append(data.distances)
    return torch.cat(ŷ), torch.cat(y), torch.cat(d)
