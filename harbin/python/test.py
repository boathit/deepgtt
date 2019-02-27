
import model, utils
import numpy as np
import imp
imp.reload(model)
imp.reload(utils)


lengths = utils.get_lengths()
dict_u, num_u = utils.get_dict_u()
dict_s1, num_s1 = utils.get_dict_s1()
dict_s2, num_s2 = utils.get_dict_s2()
dict_s3, num_s3 = utils.get_dict_s3()

dim_u = 128
dim_s1 = 64
dim_s2 = 32
dim_s3 = 16
dim_rho = 256

probrho = model.ProbRho(num_u, dim_u, dict_u, lengths,
                  num_s1, dim_s1, dict_s1,
                  num_s2, dim_s2, dict_s2,
                  num_s3, dim_s3, dict_s3,
                  200, dim_rho)

#roads = torch.randint(0, 100, size=(2, 5), dtype=torch.long)

roads = torch.tensor(np.random.choice(list(dict_u.keys()), size=(2, 5)), dtype=torch.long)

#roads_u = probrho.roads2u(roads)
#roads_s1 = probrho.roads_s_i(roads, dict_s1)
#roads_s2 = probrho.roads_s_i(roads, dict_s2)
#roads_s3 = probrho.roads_s_i(roads, dict_s3)
#roads_l = probrho.roads_length(roads)


roads.shape
rho = probrho(roads)
rho.shape

## ----------------------------------------------
dim_c = 512
probtraffic = model.ProbTraffic(1, 512, dim_c)
x = torch.rand(1, 1, 138, 148)
c = probtraffic(xs)
c.shape

## ----------------------------------------------


trips = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6], [7, 8],
                  [9, 8, 7, 6, 5], [5, 6, 7, 8, 9]])
times = np.array([3, 3, 3, 2, 5, 5])
ratios = np.random.rand(6, 2)
S = np.random.rand(2,3)

slotdata = SlotData(trips, times, ratios, S)
sd = slotdata.order_emit(1)
if sd != None:
    print(sd.trips)
    print(sd.ratios)

## -------------------------------------------------
## test the `log_prob_loss()`
t = torch.tensor([30.0, 35.0])
logμ = torch.tensor([10.4012, 13.5553], requires_grad=True)
logλ = torch.tensor([4.5233, 4.5193], requires_grad=True)
optimizer = torch.optim.Adam([logμ, logλ], lr=0.001)
for i in range(5000):
    loss = log_prob_loss(logμ, logλ, t)
    if i % 1000 == 0:
        μ = torch.exp(logμ)
        λ = torch.exp(logλ)
        print(loss.item())
        print("μ ", μ)
        print("λ ", λ)
        print("var: ", μ.pow(3) / λ)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()s

##-------------------------------------------------------
## parameter setting
lengths = utils.get_lengths()
dict_u, num_u = utils.get_dict_u()
dict_s1, num_s1 = utils.get_dict_s1()
dict_s2, num_s2 = utils.get_dict_s2()
dict_s3, num_s3 = utils.get_dict_s3()
dim_u, dim_rho, dim_c = 128, 256, 512
dim_s1 = 64
dim_s2 = 32
dim_s3 = 16

probrho = model.ProbRho(num_u, dim_u, dict_u, lengths,
                        num_s1, dim_s1, dict_s1,
                        num_s2, dim_s2, dict_s2,
                        num_s3, dim_s3, dict_s3,
                        200, dim_rho)
probtraffic = model.ProbTraffic(1, 512, dim_c)
probttime = model.ProbTravelTime(dim_rho, dim_c, 512)

optimizer_rho = torch.optim.Adam(probrho.parameters(), lr=0.001)
optimizer_traffic = torch.optim.Adam(probtraffic.parameters(), lr=0.001)
optimizer_ttime = torch.optim.Adam(probttime.parameters(), lr=0.001)


T = torch.rand(1, 1, 138, 148)
roads = torch.tensor([[62, 5337, 4906, 4907, 4909, 4910, 4911],
                      [62, 5337, 4906, 4907, 4909, 4910, 4911],
                      [1994, 1956, 1958, 3745, 1895, 1897, 1899],
                      [1994, 1956, 1958, 3745, 1895, 1897, 1899]],
                      dtype=torch.long)
t = torch.tensor([30.0, 35.0, 46.0, 47.0])
w = probrho.roads_length(roads)
l = w.sum(dim=1)
w = w / w.sum(dim=1, keepdim=True)

for i in range(200):
    rho = probrho(roads)
    c = probtraffic(T)
    #rho = torch.rand(2, 7, 256)
    #c = torch.ones(1, 512)
    logμ, logλ = probttime(rho, c, w, l)

    loss = log_prob_loss(logμ, logλ, t)
    if i % 50 == 0:
        print(loss)
        μ = torch.exp(logμ)
        λ = torch.exp(logλ)
        print("mean: ", μ)
        print("var: ", μ.pow(3) / λ)

    optimizer_rho.zero_grad()
    optimizer_traffic.zero_grad()
    optimizer_ttime.zero_grad()

    loss.backward()

    clip_grad_norm_(probrho.parameters(), 5.0)
    clip_grad_norm_(probtraffic.parameters(), 5.0)
    clip_grad_norm_(probttime.parameters(), 5.0)
    optimizer_rho.step()
    optimizer_traffic.step()
    optimizer_ttime.step()
############################################################
T = torch.rand(1, 1, 138, 148)
roads = torch.tensor([[62, 5337, 4906, 4907, 4909, 4910, 4911],
                      [62, 5337, 4906, 4907, 4909, 4910, 4911],
                      [1994, 1956, 1958, 3745, 1895, 1897, 1899],
                      [1994, 1956, 1958, 3745, 1895, 1897, 1899]],
                      dtype=torch.long)
ratios = torch.ones(roads.shape)
times = torch.tensor([30.0, 35.0, 46.0, 47.0])
model = TTime(num_u, dim_u, dict_u,
              num_s1, dim_s1, dict_s1,
              num_s2, dim_s2, dict_s2,
              num_s3, dim_s3, dict_s3,
              dim_rho, dim_c, lengths,
              hidden_size1, hidden_size2, hidden_size3,
              dropout, use_selu, device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

for i in range(200):
    logμ, logλ, mu_c, logvar_c = model(roads, ratios, T)
    loss = log_prob_loss(logμ, logλ, times)
    if i % 50 == 0:
        print(loss)
        μ = torch.exp(logμ)
        λ = torch.exp(logλ)
        print("mean: ", μ)
        print("var: ", μ.pow(3) / λ)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), .5)
    optimizer.step()
