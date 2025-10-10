# %%
import numpy as np
from copy import deepcopy
from math import exp, sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from scipy.sparse import lil_matrix
import os
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
#from functions import gaussian_mixture_field, make_heatmap_gif, adjacency_matrix, block_diag_batch
#from classes import Simulator, FireGraph, BelieverModel, DatasetGenerator, Trainer

# %%
def gaussian_mixture_field(size, n_components=None):
    '''
    Generate random topology (height distribution).
    '''
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    field = np.zeros((size, size))

    if not n_components:
        n_components = np.random.poisson(20)

    for _ in range(n_components):
        cx, cy = np.random.uniform(0, 1, 2) # random center
        sx, sy = np.random.uniform(0.01, 0.2, 2) # random covariance scale
        w = np.random.exponential(10) # weight scale

        gaussian = w * np.exp(-(((x - cx) ** 2) / (2 * sx**2) +
                                ((y - cy) ** 2) / (2 * sy**2)))
        field += gaussian

    field = (1 - (field - field.min()) / (field.max() - field.min()))*2
    return field

def make_heatmap_gif(simulator, filename="simulation.gif", cmap="plasma"):
    '''
    Create gif of simulation results.
    '''
    
    times = sorted(simulator.maps.keys())
    frames = [simulator.maps[t] for t in times]

    fig, ax = plt.subplots()

    # --- fire as background ---
    vmin, vmax = np.min(frames), np.max(frames)
    fire_img = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax)

    # --- terrain overlay ---
    terrain_img = ax.imshow(simulator.terrain, cmap="Greens", alpha=0.2)  # low alpha on top

    cbar = fig.colorbar(fire_img, ax=ax)
    cbar.set_label("Fire Intensity", rotation=270, labelpad=15)

    def update(frame):
        fire_img.set_data(frame)    
        return [fire_img, terrain_img]

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=60, blit=True
    )

    ani.save(filename, writer="pillow")
    plt.close(fig)

def adjacency_matrix(length, width):
    '''
    Sparse adjacency matrix generator for an n x n undirected graph, where each node is connected to its immediate neighbors and itself.
    '''
    N = length * width
    adj = lil_matrix((N, N), dtype=np.float32)
    directions = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (0,0)] # 8 sided
    for i in range(N):
        x, y = divmod(i, width)
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0 <= nx < length and 0 <= ny < width:
                j = nx*width + ny
                adj[i,j] = 1
    return adj.tocoo()

def block_diag_batch(adj, batch_size):
    """
    Build block-diagonal adjacency matrix for efficient processing of batches.

    This function now calls `.coalesce()` on the incoming sparse tensor so it's safe to access
    `.indices()` and `.values()` without raising "Cannot get indices on an uncoalesced tensor".
    """
    # Ensure the sparse tensor is coalesced before reading indices/values
    adj = adj.coalesce()

    N = adj.size(0)
    indices = adj.indices()  # shape (2, E)
    values = adj.values()

    device = indices.device

    offsets = torch.arange(batch_size, device=device) * N
    offsets = offsets.view(1, -1, 1)

    expanded = indices.unsqueeze(1).expand(-1, batch_size, -1)
    expanded = expanded + offsets
    expanded = expanded.permute(1, 0, 2).reshape(2, -1)

    expanded_values = values.repeat(batch_size)

    size = (N * batch_size, N * batch_size)
    # Return a coalesced sparse tensor for downstream operations
    return torch.sparse_coo_tensor(expanded, expanded_values, size=size, device=device).coalesce()


# %%
class Simulator():
    def __init__(self, size=256, wind_speed=0, wind_direction=[0,0], response_rate=0.01, response_start=20, base_spread_rate=0.01, n_components=None, decay_rate=1e-3):
        self.size = size
        self.map = np.zeros((size, size))
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.response_rate = response_rate
        self.response_start = response_start
        self.spread_rate = base_spread_rate
        self.time = 0
        self.decay_rate = decay_rate
        self.maps = {}

        self.terrain = gaussian_mixture_field(size, n_components=n_components)

    def step(self):
        new_map = deepcopy(self.map)

        for i in range(self.size):
            for j in range(self.size):
                if self.map[i, j] >= 1:
                    if np.random.rand() < self.spread_rate*self.terrain[i, j]*np.exp(-self.decay_rate * self.time) and new_map[i, j] < 5:
                        new_map[i, j] += 1
                    
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue

                            ni, nj = i + di, j + dj
                            spread_chance = self.spread_rate*self.map[i,j]

                            if 0 <= ni < self.size and 0 <= nj < self.size:
                                if self.wind_speed > 0:
                                    wind_influence = (di * self.wind_direction[0] + dj * self.wind_direction[1]) / (np.linalg.norm(self.wind_direction) + 1e-6)
                                    wind_influence *= np.random.normal(1, 0.5)


                                    if wind_influence > 0:
                                        spread_chance *= (1 + self.wind_speed * wind_influence)
                                    spread_chance *= self.terrain[ni, nj]
                                    spread_chance *= np.exp(-self.decay_rate * self.time)
                                    spread_chance = np.clip(spread_chance, 0, 1)

                                if np.random.rand() < spread_chance and new_map[ni, nj] <= new_map[i, j]:
                                    if new_map[ni, nj] < 5:
                                        if np.random.rand() <= exp(-self.time/1000):
                                            new_map[ni, nj] += 1

                                if self.time >= self.response_start and new_map[ni, nj] == 0:
                                    if np.random.rand() < 1 - exp(-(self.response_rate*(0.5+self.terrain[i,j]) * (self.time - self.response_start))):
                                        if new_map[i, j] > 0:
                                            new_map[i, j] -= 1 # Firefighting effort
                            if np.exp(-self.decay_rate * self.time) < 0.5:
                                if ni < 0 or ni >= self.size or nj < 0 or nj >= self.size:
                                    if np.random.rand() < 1 - exp(-(self.response_rate) * (self.time - self.response_start)):
                                        if new_map[i, j] > 0:
                                            new_map[i, j] -= 1 # Edge effect
                    
                else:
                    if 1 < i < self.size - 1 and 1 < j < self.size - 1:
                        neighbors_on_fire = np.sum(self.map[i-1:i+2, j-1:j+2] >= 1) - (1 if self.map[i, j] >= 1 else 0)
                        if neighbors_on_fire >= 6 and new_map[i, j] == 0:
                            new_map[i, j] += 1
        
        self.maps[self.time] = deepcopy(self.map)
        self.map = new_map
        self.time += 1

    def simulate(self):
        nodes = max(np.random.poisson(3), 1)

        x_init, y_init = np.random.randint(0, self.size, size=2)
        self.map[x_init, y_init] = max(min(np.random.poisson(3), 5), 1)

        for _ in range(nodes - 1):
            while True:
                x, y = np.random.randint(-20, 21, size=2)
                new_x, new_y = x_init + x, y_init + y

                if 0 <= new_x < self.size and 0 <= new_y < self.size:
                    if self.map[new_x, new_y] == 0:
                        self.map[new_x, new_y] = min(np.random.poisson(3), 5)
                    break
    
        while np.any(self.map > 0):
            self.step()

class FireGraph(Dataset):
    def __init__(self, length=128, width=128, path="simulation_data"):
        self.path = path
        self.length = length
        self.width = width


        adj = adjacency_matrix(self.length, self.width)
        self.adjacency_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long),
            values=torch.tensor(adj.data, dtype=torch.float32),
            size=adj.shape
        )

        self.data = []

        self.save_dir = os.path.join(self.path, f"{self.length}x{self.width}")
        os.makedirs(self.save_dir, exist_ok=True)

    def generate_data(self, topology:np.array=None, past_info:np.array=None, wind_direction:np.array=np.array([0,0]),
                         wind_speed:int=0, time:int=0, label:np.array=None):
        '''
        Generates training data from the simulation configurations and results.
        '''

        flat_topo = topology.ravel()
        flat_info = past_info[:, :, 0].ravel()
        flat_info_date = past_info[:, :, 1].ravel()
        flat_label = label.ravel()
        
        data = np.stack([
            flat_topo,
            flat_info,
            flat_info_date,
            np.full(flat_topo.shape, wind_direction[0], dtype=np.float32),
            np.full(flat_topo.shape, wind_direction[1], dtype=np.float32),
            np.full(flat_topo.shape, wind_speed, dtype=np.float32),
            np.full(flat_topo.shape, time, dtype=np.float32),
            flat_label
        ], axis=1)

        return data
    
    def save_data(self, data:np.array):
        '''
        Saves data to drive.
        '''
        np.save(os.path.join(self.save_dir, f"{time.time():.0f}.npy"), data)
    
    def generate_dataset(self, pct=0.2):
        '''
        Reads data on drive to be used in training.
        pct = percentage of data to be read from disk
        '''
        arrays = []
        for file in os.listdir(self.save_dir):
            if file.endswith(".npy"):
                arr = np.load(os.path.join(self.save_dir, file), mmap_mode='r')
                n = arr.shape[0]
                m = math.ceil(pct * n)
                idx = np.random.choice(n, size=m, replace=False)
                arrays.append(arr[idx])

        if arrays:
            self.data = np.concatenate(arrays, axis=0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        arr = self.data[idx]  # arr shape = (N, F+1)
        x = arr[:, :-1]       # (N, F)
        y = arr[:, -1]        # (N,)
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()
    

# referenced https://github.com/gordicaleksa/pytorch-GAT/blob/main/The%20Annotated%20GAT%20(Cora).ipynb for some of the code.

class BelieverModel(nn.Module):
    def __init__(self, nodes=256*256, input_features=7, num_layers=3, num_heads=10, num_features_per_head=5, num_output_classes=6, dropout=False):
        super().__init__()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False
        self.N = nodes
        self.num_heads = num_heads
        self.num_features_per_head = num_features_per_head

        self.initial_transformation = nn.Linear(input_features, num_heads * num_features_per_head, bias=False)
        nn.init.xavier_normal_(self.initial_transformation.weight)

        self.a_lefts = nn.ParameterList()
        self.a_rights = nn.ParameterList()
        self.Ws = nn.ParameterList()
        for i in range(num_layers):
            a_left = nn.Parameter(torch.zeros(size=(num_heads, num_features_per_head)))
            nn.init.xavier_uniform_(a_left)
            a_right = nn.Parameter(torch.zeros(size=(num_heads, num_features_per_head)))
            nn.init.xavier_uniform_(a_right)
            W = nn.Parameter(torch.zeros(size=(num_heads, num_features_per_head, num_features_per_head)))
            nn.init.xavier_normal_(W)

            self.a_lefts.append(a_left)
            self.a_rights.append(a_right)
            self.Ws.append(W)
        
        self.final_transformation = nn.Linear(num_heads*num_features_per_head, num_output_classes)
        nn.init.xavier_normal_(self.final_transformation.weight)
    
    def forward(self, x, adj):
        # adj = (N, N) adjacency matrix, in coo matrix format
        # x = inputs (N, F_in)

        if self.dropout:
            x = self.dropout(x)
            
        N = x.size(0)
        x = self.initial_transformation(x) # (N, F_out*H)
        x = x.view(N, self.num_heads, self.num_features_per_head) # (N, H, F_out)

        for W, a_left, a_right in zip(self.Ws, self.a_lefts, self.a_rights):
            # W = (H, F_out, F_out)
            # a_left = (H, F_out)
            # a_right = (H, F_out)

            # alpha_i,j = exp(a * [h_i||h_j] * adj[i,j]) / sum_j(exp(a * [h_i||h_j] * adj[i,j])), softmax
            # h(i') = adj * sum_j (alpha_i,j * W * h_j)
            # to simplify, we split a into 2 parts a_left and a_right, and calculate the attention scores for each of those parts, then sum up the scores only for viable pairs for computational efficiency

            h_prime = torch.einsum("nhf,hfo->nho", x, W) # (N, H, F_out) x (H, F_out, F_out) -> (N, H, F_out)

            source_scores = (h_prime * a_left).sum(-1) # elementwise product, (N, H)
            neighbor_scores = (h_prime * a_right).sum(-1) # (N, H)

            adj = adj.coalesce()
            row, col = adj.indices()
            row = row.long(); col = col.long()
            e = self.leakyrelu(source_scores[row] + neighbor_scores[col]) # (E, H) where E is the number of edges

            H =e.size(1)
            if hasattr(torch.Tensor, "scatter_reduce"):
                max_per_node = torch.zeros((N, H), device=e.device, dtype=e.dtype).scatter_reduce(
                    0, row.unsqueeze(-1).expand(-1,H), e, reduce="amax", include_self=False
                ) # maximum score among all neighbors of node i, (N, H)
            else:
                # fallback: compute max per node manually (safe but slower)
                max_per_node = torch.full((N,H), -1e9, device=e.device, dtype=e.dtype)
                for i_edge, i_node in enumerate(row):
                    max_per_node[i_node] = torch.maximum(max_per_node[i_node], e[i_edge])
            exp_e = torch.exp(e - max_per_node[row]) # for numerical stability, (E, H)

            denom = torch.zeros((N, H), device=e.device, dtype=e.dtype)
            denom.index_add_(0, row, exp_e) # summation over neighbors, (N, H)

            alpha = exp_e / (denom[row] + 1e-9) # elementwise division, (E, H)

            messages = h_prime[col] * alpha.unsqueeze(-1) # (E, H, F)

            out = torch.zeros_like(h_prime, device=h_prime.device, dtype=h_prime.dtype)
            out.index_add_(0, row, messages) # summation over neighbors again, (N, H, F)
            x = self.relu(out)
        
        x = x.reshape(N, self.num_heads * self.num_features_per_head)
        logits = self.final_transformation(x)
        return logits

class DatasetGenerator():
    def __init__(self, size=128, wind_speed=1.5, wind_direction=[1,2], response_rate=0.01, response_start=100, base_spread_rate=0.01, perturb=True):
        self.size = size
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.response_rate = response_rate
        self.response_start = response_start
        self.base_spread_rate = base_spread_rate
        self.perturb = perturb
        self.dataset = FireGraph(length=self.size, width=self.size)

    def generate(self, num_sims=100, std_dev=0.2):
        '''
        Generates simulations and saves their data to drive. Samples data randomly for initial model training.
        '''

        for i in range(num_sims):
            if self.perturb:
                
                simulator = Simulator(size=self.size, wind_speed=np.random.normal(self.wind_speed, std_dev*self.wind_speed), wind_direction=[np.random.uniform(-1, 1), np.random.uniform(-1,1)], 
                    response_rate=max(np.random.normal(self.response_rate, std_dev*self.response_rate), 0.005),
                    response_start=max(np.random.normal(self.response_start, math.floor(std_dev*self.response_start)), 0), 
                    base_spread_rate=max(np.random.normal(self.base_spread_rate, std_dev*self.base_spread_rate), 0),
                )
            else:
                simulator = Simulator(size=self.size, wind_speed=self.wind_speed, wind_direction=self.wind_direction, response_rate=self.response_rate, 
                    response_start=self.response_start, base_spread_rate=self.base_spread_rate)
                
            simulator.simulate()

            simulation_data = []
            past_info = np.zeros((self.size, self.size, 2))
            for t in simulator.maps:
                if 0.5 > np.random.rand():
                    # sample a local patch where an agent observed the previous map
                    coords = np.random.randint(0, self.size, 2)   # FIX: use grid size, not hard-coded 256
                    if t>0:
                        prev_map = simulator.maps[t-1]
                        y_min, y_max = max(0, coords[0]-5), min(self.size, coords[0]+6)
                        x_min, x_max = max(0, coords[1]-5), min(self.size, coords[1]+6)

                        past_info[y_min:y_max, x_min:x_max, 0] = prev_map[y_min:y_max, x_min:x_max]
                        past_info[y_min:y_max, x_min:x_max, 1] = 0     

                    past_info[:, :, 1] += 1

                
                data_point = self.dataset.generate_data(topology=simulator.terrain, past_info=past_info, wind_direction=np.array(simulator.wind_direction), 
                                           wind_speed=simulator.wind_speed, time=t, label=simulator.maps[t])
                simulation_data.append(data_point)
                
            self.dataset.save_data(np.array(simulation_data))

class Trainer():
    def __init__(self, size=128, model_layers=10, lr=1e-4, num_output_classes=6, model_dir=None, cnn=True):
        self.generator = DatasetGenerator(size=size)

        if cnn:
            self.model = BelieverCNN(size=size, num_output_classes=num_output_classes, encoder_layers=model_layers, hidden_channels=32)
        else:
            self.model = BelieverModel(num_layers=model_layers, num_output_classes=num_output_classes, nodes=size**2)
        
        if model_dir:
            self.model.load_state_dict(torch.load(model_dir))
            self.model_dir = model_dir
        else:
            self.model_dir = f"model/model{size}.pth"


        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        


        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

        self.loss_fn = nn.CrossEntropyLoss() # ordinal labels

    def generate(self, num_sims=100):
        '''
        Generates new simulations to be used to train.
        '''

        self.generator.generate(num_sims=num_sims)
        print(f'Generated and saved {num_sims} simulations.')

    def clear_data(self):
        dir = self.generator.dataset.save_dir

        for f in os.listdir(dir):
            if f.endswith(".npy"):
                os.remove(os.path.join(dir, f))

    def train_without_replacement(self, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
              num_sims=100, num_epochs=100):
        '''
        Generates new simulations for each epoch and clears old simulations to reduce overfitting risk.
        '''

        self.model.to(device)
        self.model.train()
        
        for epoch in range(num_epochs):
            self.generate(num_sims=num_sims)

            self.generator.dataset.generate_dataset()
            print(f"Generated the simulations for {epoch+1}/{num_epochs}.")
            train_loader = DataLoader(self.generator.dataset, batch_size=32, shuffle=True,
                                      num_workers=4, pin_memory=True, persistent_workers=True)

            total_loss = 0.0
            total_correct = 0
            total_nodes = 0

            for batch_x, batch_y in train_loader:
                B, N, F = batch_x.shape
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                flat_x = batch_x.reshape(B * N, F)
                adj_batch = block_diag_batch(self.generator.dataset.adjacency_matrix.to(device), B)

                self.optimizer.zero_grad()
                logits = self.model(flat_x, adj_batch)

                loss = self.loss_fn(
                    logits, 
                    batch_y.reshape(-1)
                ) # averaged over all nodes in all maps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                total_correct += (preds == batch_y.reshape(-1)).sum().item()
                total_nodes += batch_y.numel()

            avg_loss = total_loss / len(train_loader)
            acc = total_correct / total_nodes
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

            self.clear_data()

            self.scheduler.step()

        torch.save(self.model.state_dict(), self.model_dir)
    
    def train_with_replacement(self, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs=10):
        '''
        Retrains on past completed simulations, without generating new ones.
        '''

        self.model.to(device)
        self.model.train()
    
        for epoch in range(num_epochs):
            
            self.generator.dataset.generate_dataset()
            train_loader = DataLoader(self.generator.dataset, batch_size=32, shuffle=True,
                  num_workers=4, pin_memory=True, persistent_workers=True)
            
            print(f"Generated the dataset for epoch {epoch+1}/{num_epochs}.")

            total_loss = 0.0
            total_correct = 0
            total_nodes = 0

            for batch_x, batch_y in train_loader:
                B, N, F = batch_x.shape
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                flat_x = batch_x.reshape(B * N, F)
                adj_batch = block_diag_batch(self.generator.dataset.adjacency_matrix.to(device), B)

                self.optimizer.zero_grad()
                logits = self.model(flat_x, adj_batch)

                loss = self.loss_fn(
                    logits, 
                    batch_y.reshape(-1)
                ) # averaged over all nodes in all maps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                total_correct += (preds == batch_y.reshape(-1)).sum().item()
                total_nodes += batch_y.numel()

            avg_loss = total_loss / len(train_loader)
            acc = total_correct / total_nodes   
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

            self.scheduler.step()

        torch.save(self.model.state_dict(), self.model_dir)

    def generate_comparison_gif(self, filename="comparison.gif", cmap="plasma", 
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                new_sim = False):
        '''
        Generates a gif to compare truth, model knowledge, and model prediction as the situation evolves.
        '''

        if new_sim:
            self.generate(num_sims=1)

        dataset = self.generator.dataset
        files = [f for f in os.listdir(dataset.save_dir) if f.endswith('.npy')]
        if not files:
            raise RuntimeError(f"No simulation files found in {dataset.save_dir}")

        latest = max(files, key=lambda f: os.path.getmtime(os.path.join(dataset.save_dir, f)))
        sim_path = os.path.join(dataset.save_dir, latest)

        sim = np.load(sim_path)
        T = sim.shape[0]
        N = sim.shape[1]
        size = int(np.sqrt(N))

        terrain = sim[0, :, 0].reshape(size, size)

        past_info_seq = sim[:, :, 1].reshape(T, size, size)
        past_info_date_seq = sim[:, :, 2].reshape(T, size, size)

        truth_seq = sim[:, :, -1].reshape(T, size, size)

        self.model.to(device)
        self.model.eval()

        # Compute model predictions for each timestep
        pred_seq = []
        with torch.no_grad():
            for t in range(T):
                x_t = torch.from_numpy(sim[t, :, :-1]).float().to(device)  # (N, F)
                logits = self.model(x_t, dataset.adjacency_matrix.to(device))
                pred_map = logits.argmax(dim=1).cpu().numpy().reshape(size, size)
                pred_seq.append(pred_map)
        pred_seq = np.array(pred_seq)

        # compute vmin/vmax for fire panels (truth + model)
        vmin_fire = float(min(truth_seq.min(), pred_seq.min()))
        vmax_fire = float(max(truth_seq.max(), pred_seq.max()))

        # compute vmin/vmax for knowledge panel
        vmin_k = float(past_info_seq.min())
        vmax_k = float(past_info_seq.max())

        if vmax_k <= 0:
            knowledge_vis_scale = 1.0
        else:
            knowledge_vis_scale = max(1.0, (0.8 * (vmax_fire + 1e-9)) / (vmax_k))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        panels = ["Truth", "Knowledge", "Model Prediction"]
        fire_imgs = []
        terrain_imgs = []

        for ax, title in zip(axes, panels):
            ax.set_title(title)
            ax.axis("off")

            if title == "Knowledge":
                fire_img = ax.imshow(np.zeros((size, size)), cmap=cmap, vmin=vmin_k * knowledge_vis_scale, vmax=vmax_k * knowledge_vis_scale)
            else:
                fire_img = ax.imshow(np.zeros((size, size)), cmap=cmap, vmin=vmin_fire, vmax=vmax_fire)
            fire_imgs.append(fire_img)

            terrain_img = ax.imshow(terrain, cmap="Greens", alpha=0.3)
            terrain_imgs.append(terrain_img)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        from matplotlib import colors
        fig.colorbar(fire_imgs[0], cax=cbar_ax)

        max_frame = truth_seq.shape[0]

        def update(frame_idx):
            fire_imgs[0].set_data(truth_seq[frame_idx])

            # Knowledge
            pinfo = past_info_seq[frame_idx]
            pdate = past_info_date_seq[frame_idx]
            tau = 10.0
            fading_map = pinfo * np.exp(-pdate / tau)
            fading_vis = fading_map * knowledge_vis_scale
            fire_imgs[1].set_data(fading_vis)

            # Model predictions for this timestep
            fire_imgs[2].set_data(pred_seq[frame_idx])

            return fire_imgs + terrain_imgs

        ani = animation.FuncAnimation(fig, update, frames=max_frame, interval=60, blit=False)
        ani.save(filename, writer="pillow")
        plt.close(fig)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_dir)

# %%
class BelieverCNN(nn.Module):
    def __init__(self, size=128, input_channels=7, num_output_classes=6, encoder_layers=5, hidden_channels=32):
        super().__init__()
        self.size = size
        self.input_channels = input_channels
        self.num_output_classes = num_output_classes
        
        # input: (B, 7, H, W)
        encoders = []
        for _ in range(encoder_layers):
            encoders.extend([
                nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding='same'),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            ])
            input_channels = hidden_channels


        self.encoder = nn.Sequential(*encoders)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, num_output_classes, kernel_size=1)
        )

    def forward(self, x, adj):
        """
        x: (B*N, F) where N = size^2
        Returns: (B*N, num_output_classes)
        """
        B = x.shape[0] // (self.size ** 2)
        F = self.input_channels
        H = W = self.size

        # reshape to (B, F, H, W)
        x = x.view(B, H, W, F).permute(0, 3, 1, 2)  # (B, F, H, W)
        x = self.encoder(x)
        x = self.decoder(x)  # (B, num_output_classes, H, W)

        # flatten back to (B*N, num_output_classes)
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, self.num_output_classes)
        return x

# %%
def _worker_simulate_and_save(worker_idx, size, base_kwargs, save_dir):
    np.random.seed(int(time.time()) ^ (os.getpid() << 16) ^ worker_idx)  # reseed
    sim = Simulator(size=size,
                    wind_speed=base_kwargs.get('wind_speed', 1.5),
                    wind_direction=base_kwargs.get('wind_direction', [1,2]),
                    response_rate=base_kwargs.get('response_rate', 0.03),
                    response_start=base_kwargs.get('response_start', 100),
                    base_spread_rate=base_kwargs.get('base_spread_rate', 0.03),
                    n_components=base_kwargs.get('n_components', None),
                    decay_rate=base_kwargs.get('decay_rate', 1e-3))
    sim.simulate()

    dataset = FireGraph(length=size, width=size)
    simulation_data = []
    past_info = np.zeros((size, size, 2), dtype=np.float32)
    for t in sorted(sim.maps.keys()):
        if 0.5 > np.random.rand():
            coords = np.random.randint(0, size, 2)
            if t > 0:
                prev_map = sim.maps[t-1]
                y_min, y_max = max(0, coords[0]-5), min(size, coords[0]+6)
                x_min, x_max = max(0, coords[1]-5), min(size, coords[1]+6)
                past_info[y_min:y_max, x_min:x_max, 0] = prev_map[y_min:y_max, x_min:x_max]
                past_info[y_min:y_max, x_min:x_max, 1] = 0
            past_info[:, :, 1] += 1

        dp = dataset.generate_data(topology=sim.terrain,
                                   past_info=past_info,
                                   wind_direction=np.array(sim.wind_direction),
                                   wind_speed=sim.wind_speed,
                                   time=t,
                                   label=sim.maps[t])
        simulation_data.append(dp)

    out = np.array(simulation_data)

    # unique filename: timestamp + pid + worker idx
    fname_base = f"{int(time.time()):d}_{os.getpid()}_{worker_idx}"
    p = os.path.join(save_dir, fname_base + '.npy')
    np.save(p, out)
    return p

# %%
# Parallel simulation generator using ProcessPoolExecutor

def generate_parallel(num_sims=100, size=128, base_kwargs=None):
    if base_kwargs is None:
        base_kwargs = {}
    save_dir = os.path.join('simulation_data', f"{size}x{size}")
    os.makedirs(save_dir, exist_ok=True)
  
    available_cpus = multiprocessing.cpu_count()
    max_workers = max(1, available_cpus - 1)

    futures = []
    created = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        for i in range(num_sims):
            futures.append(exe.submit(_worker_simulate_and_save, i, size, base_kwargs, save_dir))
        for fut in as_completed(futures):
            try:
                path = fut.result()
                created.append(path)
            except Exception as e:
                print('Worker failed:', e)

    print(f'Finished. Created {len(created)} files in {save_dir}')
    return created

# %%
import shutil

def run_overfit_with_replacement(size=64, num_sims=32, num_epochs=10, lr=1e-3, device=torch.device('cpu')):

    trainer = Trainer(size=size, model_layers=32, lr=lr, num_output_classes=6, model_dir=None)
    save_dir = trainer.generator.dataset.save_dir

    if os.path.isdir(save_dir):
        print('Removing existing save_dir to ensure clean dataset:', save_dir)
        shutil.rmtree(save_dir)

    # Ensure save_dir exists again so Trainer.generate can write files
    os.makedirs(save_dir, exist_ok=True)

    print(f'Starting Trainer.train_with_replacement for {num_epochs} epoch(s) on device={device}')
    trainer.train_without_replacement(device=device, num_epochs=num_epochs, num_sims=num_sims)

    return trainer

#trainer = run_overfit_with_replacement(size=64, num_sims=64, num_epochs=10, device=torch.device('cpu'))
#trainer.generate_comparison_gif(new_sim=True)

# %%
#model_dir = "model/model128.pth"
trainer = Trainer(size=128, model_layers=32, lr=1e-3, num_output_classes=6, model_dir=None)
trainer.train_without_replacement(device=torch.device('cpu'), num_sims=64, num_epochs=20)
trainer.generate_comparison_gif(new_sim=True)


