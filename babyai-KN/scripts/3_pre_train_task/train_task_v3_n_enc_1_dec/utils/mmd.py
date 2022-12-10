import torch


def mmd_(x, y, gammas):
    gammas = gammas

    cost = torch.mean(gram_matrix(x, x, gammas=gammas))
    cost += torch.mean(gram_matrix(y, y, gammas=gammas))
    cost -= 2 * torch.mean(gram_matrix(x, y, gammas=gammas))

    if cost < 0:
        return torch.tensor(0)
    return cost


def gram_matrix(x, y, gammas):
    gammas = gammas.unsqueeze(1)
    pairwise_distances = torch.cdist(x, y, p=2.0)

    # pairwise_distances_sq = torch.square(pairwise_distances)
    pairwise_distances_sq = pairwise_distances ** 2
    tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
    tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape)
    return tmp
