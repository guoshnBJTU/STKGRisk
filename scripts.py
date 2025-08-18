# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

def shredFacts(facts, device, ds_name=None): #takes a batch of facts and shreds it into its columns
    heads      = torch.tensor(facts[:,0]).long().to(device)
    rels       = torch.tensor(facts[:,1]).long().to(device)
    tails      = torch.tensor(facts[:,2]).long().to(device)
    years = torch.tensor(facts[:,3]).float().to(device)
    months = torch.tensor(facts[:,4]).float().to(device)
    days = torch.tensor(facts[:,5]).float().to(device)
    if ds_name == "brooklyn" or ds_name == "manhatton":
        timestamps = torch.tensor(facts[:,6]).float().to(device)
        return heads, rels, tails, years, months, days, timestamps
    else:
        return heads, rels, tails, years, months, days
