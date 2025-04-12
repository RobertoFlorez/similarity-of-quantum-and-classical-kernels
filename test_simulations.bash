#Here we send test simulations to check if that the code is working properly.

echo "Running classical simulations"
py -3 -m classical_performance 0 0 1 #test classic simulations
wait 
mv ./data/results/performance_classical_0_0 ./data/results/plasticc_test_classical
echo "Calculating quantum kernel simulations"
py -3 -m kernel_main 0 0 1
wait
echo "Training and running quantum simulations"
py -3 -m quantum_performance 0 0 1
wait
echo "Calculating differences between classical and quantum simulations"
py -3 -m all_difference_calculations 0 0 plasticc_test
wait
