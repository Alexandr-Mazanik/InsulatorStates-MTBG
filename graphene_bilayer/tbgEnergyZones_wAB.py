import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

from scipy.sparse import coo_matrix, csr_matrix
from joblib import Parallel, delayed
from tqdm import tqdm


N_JOBS = 6             # number of processes 

w_AA = 0               # meV  -- interlayer hopping energy (AA bilayer stacking)
w_AB = 110             # meV  -- interlayer hopping energy (AB bilayer stacking)
t = 2730               # meV  -- intralayer hopping energy

# alpha = 0.585663558388 - exact first magic angle

alpha = 0.586          # 1 / theta (see Tarnopolsky et al. (2019))
cutoff = 10            # number of shells to include (more for better accuracy)

N = 400                # density of k points          
num_bands_show = 10    # number of zones displayed


vF = 3 * t / 2
phi = 2 * np.pi / 3
kD = 4 * np.pi / (3 * np.sqrt(3))
theta = 2 * np.arcsin(w_AB / (2 * vF * kD * alpha))
k_moire = 2 * kD * np.sin(theta / 2)


def main():

  print("===========================================")
  print("--> Moire energy bands calculation (w_AA != w_AB):")
  print(f"--> theta: {np.round(np.degrees(theta), 3)} degrees")
  print(f"    cutoff: {cutoff}")
  print(f"    w_AA = {w_AA} meV, w_AB = {w_AB} meV")
  print(f"    t = {t} meV")
  print(f"    alpha: {np.round(alpha, 3)}\n")

  brillouin_path = moire_brillouin_path()
  bands = calculate_bands(brillouin_path)

  plot_bands(bands, brillouin_path)
  print("===========================================")


class State:
  def __init__(self, n, m, layer_idx):
    self.n = n
    self.m = m
    self.layer = layer_idx
    self.basis_position = None


  def set_basis_position(self, n):
    self.basis_position = n 


  def get_neighbors(self):
    return [State(self.n, self.m, -self.layer), 
            State(self.n + self.layer, self.m, -self.layer),
            State(self.n, self.m - self.layer, -self.layer)]


  def state_momentum(self):

    b1 = k_moire * np.array([np.sqrt(3) / 2, 3 / 2])
    b2 = k_moire * np.array([np.sqrt(3) / 2, - 3 / 2])

    kx, ky = self.n * b1 + self.m * b2

    if self.layer == -1:
      ky -= k_moire

    return (kx, ky)


  def __eq__(self, other):
    if isinstance(other, State):
      return self.n == other.n and self.m == other.m and self.layer == other.layer
    return NotImplemented


def calculate_bands(brillouin_path):

  states = basis_states()
  dim = len(states)

  bands = [[] for _ in range(2 * dim)]

  time_start = time.time()

  results = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(compute_eigs)(k_cartesian, states, dim)
    for k_cartesian in tqdm(zip(brillouin_path[0], brillouin_path[1]), total=3*N, desc="Computing...")
  )

  bands = [[] for _ in range(2 * dim)]
  for eigen_val in results:
    for i in range(2 * dim):
      bands[i].append(eigen_val[i])

  time_end = time.time()
  print("\n--> The time of execution is :", round(time_end - time_start), "s")

  bands = [[] for _ in range(2 * dim)]
  
  for eigen_val in results:
    for i in range(2 * dim):
      bands[i].append(eigen_val[i])

  return [np.array(band) for band in bands]


def basis_states():
  zero_state = State(0, 0, 1)
  zero_state.set_basis_position(0)

  states = [ zero_state ]
  shells = [ [ zero_state ] ]
  
  state_position = 1
  for k in range(cutoff):
    shells.append([])
    for state in shells[-2]:
      for neighbor in state.get_neighbors():
        if not any(neighbor == _ for  _ in states):
          neighbor.set_basis_position(state_position)

          shells[-1].append(neighbor)
          states.append(neighbor)

          state_position += 1 
  
  return states


def compute_eigs(k_cartesian, states, dim):
  H = H_n(k_cartesian, states, dim)
  assert np.allclose(H, H.conj().T, atol=1e-12)

  eigen_val = np.linalg.eigvalsh(H)
    
  return eigen_val


def H_n(k_cartesian, states, dim):
  kx, ky = k_cartesian

  blocks = []
  positions = []

  state_map = { (s.n, s.m, s.layer) : s.basis_position for s in states }
  
  # diagonal term
  for state in states:
    i = state.basis_position

    kx_state, ky_state = state.state_momentum()
    blocks.append(h_k((kx - kx_state, ky - ky_state), state.layer * theta / 2))
    positions.append((i, i))
  
  # off-diagonal term
  for state in states:
    i = state.basis_position
    
    neighbor_keys = [
      (state.n, state.m, -state.layer),
      (state.n + state.layer, state.m, -state.layer),
      (state.n, state.m - state.layer, -state.layer)
    ]
    
    for key in neighbor_keys:
      if key in state_map:
        j = state_map[key]
        if j > i:                       
          if key[0] != state.n:
            coupling = T_tr()
          elif key[1] != state.m:
            coupling = T_tl()
          else:
            coupling = T_b()
          blocks.append(coupling)
          positions.append((i, j))
          blocks.append(dagger(coupling))
          positions.append((j, i))

  H = (create_block_matrix(positions, blocks, dim)).toarray()

  return H


def cartesian_to_polar(vec_x, vec_y):
  magnitude = np.hypot(vec_x, vec_y)
  theta = np.arctan2(vec_y, vec_x)
  
  return (magnitude, theta)


def dagger(matrix):
  return matrix.conj().T


def h_k(k_cartesian, theta_layer):
  kx, ky = k_cartesian
  k, theta_k = cartesian_to_polar(kx, ky)

  return - vF * k * np.array([[0, np.exp(1j * (theta_k - theta_layer))],
   [np.exp(-1j * (theta_k - theta_layer)), 0]], dtype=complex)


def T_b():
  return np.array([[w_AA, w_AB], [w_AB, w_AA]], dtype=complex)


def T_tr():
  return np.array([[w_AA * np.exp(-1j * phi), w_AB],
   [w_AB * np.exp(1j * phi), w_AA * np.exp(-1j * phi)]], dtype=complex)


def T_tl():
  return np.array([[w_AA * np.exp(1j * phi), w_AB],
   [w_AB * np.exp(-1j * phi), w_AA * np.exp(1j * phi)]], dtype=complex)


def create_block_matrix(block_coords, block_values, dim):
    size = 2 * dim
    rows, cols, data = [], [], []

    for (i, j), block in zip(block_coords, block_values):
        row_start, col_start = 2 * i, 2 * j
        for bi in range(2):
            for bj in range(2):
                rows.append(row_start + bi)
                cols.append(col_start + bj)
                data.append(block[bi, bj])

    mat = coo_matrix((data, (rows, cols)), shape=(size, size), dtype=complex)
    return mat.tocsr()


def moire_brillouin_path():

  k_KG_x = np.zeros(N)
  k_KG_y = np.linspace(-k_moire, k_moire, N)
  k_GG_x = np.arange(0, - k_moire * np.sqrt(3) / 2, - k_moire * np.sqrt(3) / 2 / N)
  k_GG_y = k_GG_x * np.sqrt(3) + k_moire
  k_GK_x = k_GG_x[::-1]
  k_GK_y = - k_GK_x / np.sqrt(3) - k_moire

  k_brillouin_x = np.concatenate([k_KG_x, k_GG_x, k_GK_x])
  k_brillouin_y = np.concatenate([k_KG_y, k_GG_y, k_GK_y])

  return k_brillouin_x, k_brillouin_y


def cumulative_k_dist(k_path):
  kx, ky = k_path

  N = kx.shape[0]

  dists = [0]
  for i in range(1, N):
    dx = kx[i] - kx[i-1]
    dy = ky[i] - ky[i-1]
    d = np.sqrt(dx**2 + dy**2)
    dists.append(dists[-1] + d)

  return np.array(dists)


def plot_bands(bands, brillouin_path):
  def central_band_indices(m, p):
    start = (m - p) // 2
    return range(start, start + p)

  k_points = cumulative_k_dist(brillouin_path)
  colors = ['orange', 'purple', 'green', 'blue', 'red']
  colors = colors + colors[::-1]

  high_symmetry_indices = [0, int(N / 2) - 1, N - 1, 2 * N - 1, 3 * N - 1]
  high_symmetry_labels = ['K', 'K', 'G', 'G', 'K']

  fig, ax = plt.subplots(figsize=(3.5, 6))

  color_ind = 0
  for i in central_band_indices(len(bands), num_bands_show):
    ax.plot(k_points, bands[i], color=colors[color_ind % len(colors)], linewidth=2)
    color_ind += 1

  for idx in high_symmetry_indices:
    x = k_points[idx]
    ax.axvline(x=x, color='gray', linestyle='--', linewidth=1)

  ax.set_xticks([k_points[i] for i in high_symmetry_indices])
  ax.set_xticklabels(high_symmetry_labels, fontsize=10)
  ax.set_ylabel("Energy", fontsize=12)
  ax.set_title(f"theta = {np.round(np.degrees(theta), 3)}; alpha = {np.round(alpha, 3)}, \n cutoff = {cutoff}", fontsize=12)

  for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)

  ax.grid(False)

  plt.tight_layout()
  
  filename = f"bands{num_bands_show}_theta{int(np.degrees(theta) * 100)}_wAA_{w_AA}_wAB_{w_AB}_cutoff{cutoff}.pdf"
  plt.savefig(filename)

  print("--> Export: ", filename, '\n')
    

if __name__ == "__main__":

    main()
