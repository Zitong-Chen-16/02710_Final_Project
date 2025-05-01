with open("parameters_mixed.in", "w") as f:
    for i in range(19):
        f.write(f"../test_{i}.txt results/geo_normal/test_{i}_mu0.001_s0.1 0.001 0.1\n")

print("Parameters written to parameters_test.in")
