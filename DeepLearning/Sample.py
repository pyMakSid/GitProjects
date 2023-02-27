import numpy as np 

fir_wht = np.array([
    [0.1, 0.2, -0.1],
    [-0.1, 0.1, 0.9],
    [0.1, 0.4, 0.1]])

sec_wht = np.array([
    [0.3, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1]])

weights = [fir_wht, sec_wht]

def neural_network(input, weights):
    mid = input.dot(weights[0])
    pred = mid.dot(weights[1])
    return pred

ntoe = [8.5, 9.5, 9.9, 9]
nwins = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1]

i = 0

input =np.array([ntoe[i], nwins[i], nfans[i]])
result = neural_network(input, weights)
print(result)