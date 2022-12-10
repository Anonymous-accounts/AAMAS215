import os
import torch

from .. import utils


def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)


def get_model_path_name(model_name, name):
    return os.path.join(get_model_dir(model_name), name)


def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "acmodel.pt")


def load_model(model_name, raise_not_found=True):
    # import pdb
    # pdb.set_trace()
    path = get_model_path(model_name)
    try:
        model = torch.load(path)
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))


def save_model(acmodel, query_encoder, att, model_name):
    # import pdb
    # pdb.set_trace()
    path = get_model_path_name(model_name, "acmodel.pt")
    utils.create_folders_if_necessary(path)
    torch.save(acmodel, path)
    
    path = get_model_path_name(model_name, "att.pt")
    utils.create_folders_if_necessary(path)
    torch.save(att, path)
    
    path = get_model_path_name(model_name, "query_encoder.pt")
    utils.create_folders_if_necessary(path)
    torch.save(query_encoder, path)
