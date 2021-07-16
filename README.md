# Multiple Traveling Salesman Problem
A brute force solution to the MTSP problem.

## Requirements
```shell
pip3 install -r requirements.txt
```
## Run
```py
python3 src/main.py examples/small/small_input.json examples/small/example_small_output.json
python3 src/main.py examples/medium/medium_input.json examples/medium/example_small_output.json
```

## How To Use
```py
from main import MTSPBruteForce
my_solver = MTSPBruteForce(INPUT)
solution = my_solver.run()
f = open(OUTPUT, "w")
f.write(solution)
f.close()
print(f"Solution is written to the {OUTPUT}:\n{solution}")
```
