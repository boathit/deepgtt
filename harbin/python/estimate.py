import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_utils import DataLoader
from model import *
import os, argparse, time

## python python-model/estimate.py -testpath travel-model/data-7min/testdata -model best-model.pt

parser = argparse.ArgumentParser(description="estimate.py")

parser.add_argument("-testpath", help="Path to test data",
    default="/home/xiucheng/data-backup/bigtable/2015-taxi/data/testdata")

parser.add_argument("-model", default="best-model.pt")

parser.add_argument("-batch_size", type=int, default=128,
    help="The batch size")

args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    logz = 0.5*logλ - 1.5*torch.log(t)
    return expt+logz

def log_prob_loss(logμ, logλ, t):
    """
    logμ, logλ (batch_size, ): the parameters of IG
    t (batch_size, ): the travel time
    ---
    Return the average loss of the log probability of a batch of data
    """
    logpdf = log_pdf(logμ, logλ, t)
    return torch.mean(-logpdf)

def trip2logIG(probrho, trips, ratios, S):
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

def load_model(fname):
    model = torch.load(fname)
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

def validate(num_iterations, probrho,
                             probtraffic,
                             probttime):
    total_loss, total_mse, total_l1 = 0.0, 0.0, 0.0
    for _ in range(num_iterations):
        data = test_dataloader.order_emit(args.batch_size)
        road_lens = probrho.roads_length(data.trips, data.ratios)
        l = road_lens.sum(dim=1) # the whole trip lengths
        w = road_lens / road_lens.sum(dim=1, keepdim=True) # road weights
        rho = probrho(data.trips)
        c, mu_c, logvar_c = probtraffic(data.S.to(device))
        logμ, logλ = probttime(rho, c, w, l)
        times = data.times.to(device)
        loss = log_prob_loss(logμ, logλ, times)
        total_loss += loss.item() * data.trips.shape[0]
        total_mse += F.mse_loss(torch.exp(logμ), times).item() * data.trips.shape[0]
        total_l1 += F.l1_loss(torch.exp(logμ), times).item() * data.trips.shape[0]
    mean_loss = total_loss / np.sum(test_slot_size)
    mean_mse = total_mse / np.sum(test_slot_size)
    mean_l1 = total_l1 / np.sum(test_slot_size)
    print("Testing Loss {0:.4f} MSE {1:.4f} L1 {2:.4f}".format(mean_loss, mean_mse, mean_l1))

    return mean_loss, mean_mse

probrho, probtraffic, probttime = load_model(args.model)
probrho.eval()
probtraffic.eval()
probttime.eval()

testfiles = list(filter(lambda x:x.endswith(".h5"),
                        sorted(os.listdir(args.testpath))))
test_dataloader = DataLoader(args.testpath)
print("Loading the testing data...")
test_dataloader.read_files(testfiles)
test_slot_size = np.array(list(map(lambda s:s.ntrips, test_dataloader.slotdata_pool)))
print("There are {} trips in total".format(test_slot_size.sum()))
test_num_iterations = int(np.ceil(test_slot_size/args.batch_size).sum())

tic = time.time()
with torch.no_grad():
    validate(test_num_iterations, probrho, probtraffic, probttime)
print("Time passed: {} seconds".format(time.time() - tic))
