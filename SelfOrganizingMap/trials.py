import numpy as np
from time import time
from BetterSOM import BetterSOM
from FileIO import FileIO
from LearningSchedule import LearningSchedule
from SelfOrganizingMap import SelfOrganizingMap
from EvaluationMetrics import \
    NeuronUtilization, \
    QuantizationError, \
    TopographicError, \
    TopologicalProduct, \
    TopographicFunction


verbose = False

ntrials = int(1e1)

data_dimension = 3

data = FileIO.read_lrn("./../Data/GolfBall.lrn")

# ecoli_txt = FileIO.read_file("./../Data/ecoli.data.txt").split("\n")
# data = {line_index: np.array(ecoli_txt[line_index].split()[1:len(ecoli_txt[line_index].split()) - 1]).astype(np.float) for line_index in range(len(ecoli_txt))}

# class_map = FileIO.read_cls("./../Data/Lsun.cls")
class_map = {}

lattice_dimension = [22, 22]

prototype_dimension = data_dimension

nsom_num_iter = int(10e3)

ksom_num_iter = nsom_num_iter

alpha_sched = LearningSchedule(dict({1000: 0.5, 2500: 0.2, 6000: 0.1, 10000: 0.01}))

sigma_sched = LearningSchedule(dict({1000: 11, 2500: 6, 6000: 3, 10000: 1}))

ksom_results = \
    list([
        list([
            "Trial Index",
            "Training Time (s)",
            "Neuron Utilization",
            "Quantization Error",
            "Topographic Error"
        ])
    ])

nsom_results = \
    list([
        list([
            "Trial Index",
            "Training Time (s)",
            "Neuron Utilization",
            "Quantization Error",
            "Topographic Error"
        ])
    ])

t0 = time()

for trial_index in range(ntrials):

    print("Running Trial " + str(trial_index) + "...\n")

    initial_prototype_matrix = np.random.rand(*(list(lattice_dimension) + list([prototype_dimension])))

    ksom = \
        SelfOrganizingMap(
            lattice_dimension=lattice_dimension,
            prototype_dimension=prototype_dimension,
            prototype_matrix=initial_prototype_matrix
        )

    ksom_t0 = time()

    ksom.train(data, alpha_sched, sigma_sched, ksom_num_iter)

    ksom_t1 = time()

    ksom_training_time = ksom_t1 - ksom_t0

    ksom_nu = NeuronUtilization.compute(ksom, data)

    ksom_qe = QuantizationError.compute(ksom, data)

    ksom_te = TopographicError.compute(ksom, data)

    # ksom_tp = TopologicalProduct.compute(ksom)

    ksom_results.append(list([trial_index, ksom_training_time, ksom_nu, ksom_qe, ksom_te]))

    nsom = \
        BetterSOM(
            lattice_dimension=lattice_dimension,
            prototype_dimension=prototype_dimension,
            prototype_matrix=initial_prototype_matrix
        )

    nsom_t0 = time()

    nsom.train(data, alpha_sched, sigma_sched, nsom_num_iter)

    nsom_t1 = time()

    nsom_training_time = nsom_t1 - nsom_t0

    nsom_nu = NeuronUtilization.compute(nsom, data)

    nsom_qe = QuantizationError.compute(nsom, data)

    nsom_te = TopographicError.compute(nsom, data)

    # nsom_tp = TopologicalProduct.compute(nsom)

    nsom_results.append(list([trial_index, nsom_training_time, nsom_nu, nsom_qe, nsom_te]))

    print("Trial " + str(trial_index) + " Completed!\n")

t1 = time()

KSOM_RESULTS_FILENAME = "KSOM Results (" + str(ntrials) + " trials)"

NSOM_RESULTS_FILENAME = "NSOM Results (" + str(ntrials) + " trials)"

FileIO.write_csv(csv_filename=KSOM_RESULTS_FILENAME, data_matrix=ksom_results)

FileIO.write_csv(csv_filename=NSOM_RESULTS_FILENAME, data_matrix=nsom_results)

print("Total time taken to run trials: ", str(t1 - t0), "seconds\n")
