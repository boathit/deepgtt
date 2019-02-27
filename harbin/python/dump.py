
def log_loss(mu, var, t, l):
    logz = 0.5*torch.log(l) + 1.5*torch.log(mu) + 0.5*torch.log(var) - 1.5*torch.log(t)
    expt = 0.5*t/(l*mu*var) - 1/(mu.pow(2)*var) + 0.5*l/(var*mu.pow(3)*t)
    #print("expt: {}\t -logz: {}".format(expt, -logz))
    return torch.mean(expt-logz)

def log_prob_loss(logμ, logλ, t):
    """
    logμ, logλ (batch_size, ): the parameters of IG
    t (batch_size, ): the travel time
    ---
    Return the average loss of the log probability of a batch of data
    """
    eps = 0
    μ = torch.exp(logμ)
    expt = 0.5 * torch.exp(logλ)*torch.pow(t-μ,2) / (μ.pow(2)*t+eps)
    logz = -0.5*logλ + 1.5*torch.log(t)
    #print("expt: {}\t logz: {}".format(expt, logz))
    loss = torch.mean(expt + logz)
    return loss

conv_layers = [
    nn.Conv2d(n_in, 32, (4, 4), stride=2, padding=1),
    nn.LeakyReLU(0.15, inplace=True),
    nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.15, inplace=True),
    nn.Conv2d(64, 128, (4, 4), stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.15, inplace=True),
    nn.Conv2d(128, 128, (4, 4), stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.15, inplace=True),
    nn.Conv2d(128, 128, (4, 4), stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU()
]
self.f1 = nn.Sequential(*conv_layers)
self.f2 = MLP2(128*4*4, hidden_size, dim_c, dropout, use_selu)


conv_layers = [
    nn.Conv2d(n_in, 32, (4, 4), stride=2, padding=1),
    nn.LeakyReLU(0.15, inplace=True),
    nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.15, inplace=True),
    nn.Conv2d(64, 128, (4, 4), stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.15, inplace=True),
    nn.Conv2d(128, 256, (4, 4), stride=2, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.AvgPool2d(3, padding=1)
]
self.f1 = nn.Sequential(*conv_layers)
self.f2 = MLP2(256*3*3, hidden_size, dim_c, dropout, use_selu)
