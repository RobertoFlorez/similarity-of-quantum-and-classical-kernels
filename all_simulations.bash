#Here we explicitly write all the simulations run in the paper.


##################
#Classical simulations
##################

py -3 -m classical_performance 0 1 1 #plasticc classic simulations
wait
py -3 -m classical_performance 0 5 1 #kMNIST28 classic simulations
wait
py -3 -m classical_performance 0 55 1 #hidden-manifold classic simulations
wait

#rename the folders

mv  ./data/results/performance_classical_0_1 ./data/results/plasticc_classical
mv  ./data/results/performance_classical_0_5 ./data/results/kMNIST28_classical
mv  ./data/results/performance_classical_0_55 ./data/results/hidden-manifold_classical



##################
#Quantum simulations
##################

##################
#plasticc
##################

py -3 -m kernel_main 0 1 1
wait
py -3 -m quantum_performance 0 1 1
wait
py -3 -m all_difference_calculations 0 1 plasticc
wait

py -3 -m kernel_main 0 3 1
wait
py -3 -m quantum_performance 0 3 1
wait
py -3 -m all_difference_calculations 0 3 plasticc
wait

py -3 -m kernel_main 0 4 1
wait
py -3 -m quantum_performance 0 4 1
wait
py -3 -m all_difference_calculations 0 4 plasticc
wait

py -3 -m kernel_main 0 25 1
wait
py -3 -m quantum_performance 0 25 1
wait
py -3 -m all_difference_calculations 0 25 plasticc
wait

py -3 -m kernel_main 0 30 1
wait
py -3 -m quantum_performance 0 30 1
wait
py -3 -m all_difference_calculations 0 30 plasticc
wait

py -3 -m kernel_main 0 9 1
wait
py -3 -m quantum_performance 0 9 1
wait
py -3 -m all_difference_calculations 0 9 plasticc
wait

py -3 -m kernel_main 0 11 1
wait
py -3 -m quantum_performance 0 11 1
wait
py -3 -m all_difference_calculations 0 11 plasticc
wait

py -3 -m kernel_main 0 12 1
wait
py -3 -m quantum_performance 0 12 1
wait
py -3 -m all_difference_calculations 0 12 plasticc
wait

py -3 -m kernel_main 0 26 1
wait
py -3 -m quantum_performance 0 26 1
wait
py -3 -m all_difference_calculations 0 26 plasticc
wait

py -3 -m kernel_main 0 32 1
wait
py -3 -m quantum_performance 0 32 1
wait
py -3 -m all_difference_calculations 0 32 plasticc
wait

##################
#kMNIST28 
##################

py -3 -m kernel_main 0 5 1
wait
py -3 -m quantum_performance 0 5 1
wait
py -3 -m all_difference_calculations 0 5 kMNIST28
wait

py -3 -m kernel_main 0 7 1
wait
py -3 -m quantum_performance 0 7 1
wait
py -3 -m all_difference_calculations 0 7 kMNIST28
wait

py -3 -m kernel_main 0 8 1
wait
py -3 -m quantum_performance 0 8 1
wait
py -3 -m all_difference_calculations 0 8 kMNIST28
wait

py -3 -m kernel_main 0 27 1
wait
py -3 -m quantum_performance 0 27 1
wait
py -3 -m all_difference_calculations 0 27 kMNIST28
wait

py -3 -m kernel_main 0 31 1
wait
py -3 -m quantum_performance 0 31 1
wait
py -3 -m all_difference_calculations 0 31 kMNIST28
wait

py -3 -m kernel_main 0 13 1
wait
py -3 -m quantum_performance 0 13 1
wait
py -3 -m all_difference_calculations 0 13 kMNIST28
wait

py -3 -m kernel_main 0 15 1
wait
py -3 -m quantum_performance 0 15 1
wait
py -3 -m all_difference_calculations 0 15 kMNIST28
wait

py -3 -m kernel_main 0 16 1
wait
py -3 -m quantum_performance 0 16 1
wait
py -3 -m all_difference_calculations 0 16 kMNIST28
wait

py -3 -m kernel_main 0 28 1
wait
py -3 -m quantum_performance 0 28 1
wait
py -3 -m all_difference_calculations 0 28 kMNIST28
wait

py -3 -m kernel_main 0 33 1
wait
py -3 -m quantum_performance 0 33 1
wait
py -3 -m all_difference_calculations 0 33 kMNIST28
wait

##################
#hidden-manifold
##################

py -3 -m kernel_main 0 55 1
wait
py -3 -m quantum_performance 0 55 1
wait
py -3 -m all_difference_calculations 0 55 hidden-manifold
wait

py -3 -m kernel_main 0 60 1
wait
py -3 -m quantum_performance 0 60 1
wait
py -3 -m all_difference_calculations 0 60 hidden-manifold
wait

py -3 -m kernel_main 0 56 1
wait
py -3 -m quantum_performance 0 56 1
wait
py -3 -m all_difference_calculations 0 56 hidden-manifold
wait

py -3 -m kernel_main 0 61 1
wait
py -3 -m quantum_performance 0 61 1
wait
py -3 -m all_difference_calculations 0 61 hidden-manifold
wait

py -3 -m kernel_main 0 57 1
wait
py -3 -m quantum_performance 0 57 1
wait
py -3 -m all_difference_calculations 0 57 hidden-manifold
wait

py -3 -m kernel_main 0 62 1
wait
py -3 -m quantum_performance 0 62 1
wait
py -3 -m all_difference_calculations 0 62 hidden-manifold
wait

py -3 -m kernel_main 0 59 1
wait
py -3 -m quantum_performance 0 59 1
wait
py -3 -m all_difference_calculations 0 59 hidden-manifold
wait

py -3 -m kernel_main 0 64 1
wait
py -3 -m quantum_performance 0 64 1
wait
py -3 -m all_difference_calculations 0 64 hidden-manifold
wait

py -3 -m kernel_main 0 58 1
wait
py -3 -m quantum_performance 0 58 1
wait
py -3 -m all_difference_calculations 0 58 hidden-manifold
wait

py -3 -m kernel_main 0 63 1
wait
py -3 -m quantum_performance 0 63 1
wait
py -3 -m all_difference_calculations 0 63 hidden-manifold
wait
