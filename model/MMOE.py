import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class MMOE(nn.Module):
    def __init__(self, input_size, num_experts, experts_out, experts_hidden):
        super(MMOE, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for _ in range(self.num_experts)])
        self.w_gates = nn.Parameter(torch.randn(input_size, num_experts), requires_grad=True)

    def forward(self, x):
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o, dim=0)

        gates_o = [self.softmax(x @ self.w_gates)]

        final_output = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        final_output = [torch.sum(ti, dim=0) for ti in final_output]
        # print("MMOE:", final_output[0].shape)

        return final_output[0]