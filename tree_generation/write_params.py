from pathlib import Path

root_folder = Path('../')
graph_dir = root_folder / Path('graphs/mixed/') 
save_dir = root_folder / Path('results/trees/mixed/')
save_dir.mkdir(parents=True, exist_ok=True)

with open("parameters.in", "w") as f:
    for file_path in list(graph_dir.glob('*.txt')):
        graph_name = file_path.name.split('.')[0]
        output_line = f"{file_path} {save_dir}/{graph_name}_mu0.001_s0.1 0.001 0.1\n"
        f.write(output_line)
        
print("Parameters written to parameters.in")