# Digital VLSI Design - Model Based Project
- Implementation of 5 different Machine Learning Models to predict the Leakage and Delay values of a given Technology for a specific circuit.

#### Pooja Srinivas - 20171403
#### Swapnil Pakhare - 20161199
#### Ayush Anand - 20161085
#### Mayank Taluja - 20161102



## RUN:
- First we need to convert all the .xlsx files to .csv format.
- To convert the files run xlsx2csv.py
- Create a new folder for a specific ciruit with all the delay and leakage files.
- Implemented for NAND2, NAND3, AND3, XOR2, NOR2 and NOR3.
- Run the python file.

```bash
python xlsx2csv.py
python model_based_project.py
```

# Digital VLSI Design - Algorithm Based Project
- Implementation of Modified Krill Herd Optimization Algorithm using Focus Group Idea for Multi Objectve Delay, Power Optimization.

## RUN:
- First we need to connect to the server to access all the netlists.
- Server IP address : 10.4.24.18.
- User name: DVD1_model.
- Navigate to the folder called 'Netlists'.
- Our Algorithm is implemented and tested on a full adder.
- Create a new folder (for a new netlist) or use the existing one 'v4' for a full adder.
- Copy all the .sp and .st files into the folder and create extra copies of these files as temporary files to update the values according to the model and reuse the original files whenever necessary.
- Run the python file in this folder.
- For Fulladder navigate to ~/v4. Run the .py file in the folder.
 
```bash
python kh_vlsi2.py
