echo 'Starting!\n' &>> status.txt
python3 automated_generator.py(1000, 100000, 0.2, 25, 1000, weak)
wait
echo '###--->1'
echo 'Big Sparse Weak Done!\n' &>> status.txt
wait
python3 automated_generator.py(1000, 100000, 0.2, 25, 1000, strong)
wait
echo '###--->2'
echo 'Big Sparse Strong Done!\n' &>> status.txt
wait
python3 automated_generator.py(1000, 100000, 0.2, 25, 1000, strong2)
wait
echo '###--->3'
echo 'Big Sparse Strong2 Done!\n' &>> status.txt
wait
python3 automated_generator.py(1000, 100000, 0.8, 50, 1000, weak)
wait
echo '###--->4'
echo 'Big Dense Weak Done!\n' &>> status.txt
wait
python3 automated_generator.py(1000, 100000, 0.8, 50, 1000, strong)
wait
echo '###--->5'
echo 'Big Dense Strong Done!\n' &>> status.txt
wait
python3 automated_generator.py(1000, 100000, 0.8, 50, 1000, strong2)
wait
echo '###--->6'
echo 'Big Dense Strong2 Done!\n' &>> status.txt
wait
python3 automated_generator.py(100, 1000, 0.2, 10, weak)
wait
echo '###--->7'
echo 'Small Sparse Weak Done!\n' &>> status.txt
wait
python3 automated_generator.py(100, 1000, 0.2, 10, strong)
wait
echo '###--->8'
echo 'Small Sparse Strong Done!\n' &>> status.txt
wait
python3 automated_generator.py(100, 1000, 0.2, 10, strong2)
wait
echo '###--->9'
echo 'Small Sparse Strong2 Done!\n' &>> status.txt
wait
python3 automated_generator.py(100, 1000, 0.8, 30, weak)
wait
echo '###--->10'
echo 'Small Dense Weak Done!\n' &>> status.txt
wait
python3 automated_generator.py(100, 1000, 0.8, 30, strong)
wait
echo '###--->11'
echo 'Small Dense Strong Done!\n' &>> status.txt
wait
python3 automated_generator.py(100, 1000, 0.8, 30, strong2)
wait
echo '###--->12'
echo 'Small Dense Strong2 Done!\n' &>> status.txt
echo 'ALL DONE!' &>> status.txt
