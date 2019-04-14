import numpy as np
from FileIO import FileIO


ntrials = int(1e2)

KSOM_RESULTS_FILENAME = "KSOM Results (" + str(ntrials) + " trials)"

NSOM_RESULTS_FILENAME = "NSOM Results (" + str(ntrials) + " trials)"

ksom_results = np.array([np.array([float(element) for element in row]) for row in FileIO.read_csv(csv_filename=KSOM_RESULTS_FILENAME)[1:]])

nsom_results = np.array([np.array([float(element) for element in row]) for row in FileIO.read_csv(csv_filename=NSOM_RESULTS_FILENAME)[1:]])

print("\n")
print("Training Time (s):", str(np.mean(ksom_results[:, 1])), "vs", str(np.mean(nsom_results[:, 1])), "\n")
print("Neuron Utilization:", str(np.mean(ksom_results[:, 2])), "vs", str(np.mean(nsom_results[:, 2])), "\n")
print("Quantization Error:", str(np.mean(ksom_results[:, 3])), "vs", str(np.mean(nsom_results[:, 3])), "\n")
print("Topographic Error:", str(np.mean(ksom_results[:, 4])), "vs", str(np.mean(nsom_results[:, 4])), "\n")
# print("Topological Product:", str(np.mean(np.abs(ksom_results[:, 5]))), "vs", str(np.mean(np.abs(nsom_results[:, 5]))), "\n")
