def merge(
    input_app_filenames, 
    input_app_rp_filenames,
    output_app_filename, 
    output_app_rp_filename
):
    
    nodes_counts = []
    arcs = []
    for filename in input_app_filenames:
        with open(filename) as f: 
            nodes_counts.append(int(f.readline()))
            arcs.append(f.readline().strip())

    with open(output_app_filename, "w") as f:

        f.write(f"{sum(nodes_counts)}\n")

        original_arcs = [arcs[0].split()]

        for i in range(1,len(arcs)):
            shift = sum(nodes_counts[:i])
            tmp_arcs = arcs[i].split()
            for j in range(len(tmp_arcs)):
                src,dest = tmp_arcs[j].split(',')
                tmp_arcs[j] = f"{int(src)+shift},{int(dest)+shift}"
            original_arcs.append(tmp_arcs)
            arcs[i] = ' '.join(tmp_arcs)

        arcs = ' '.join(arcs)
        f.write(f"{arcs}")

    # -------------------------------------------------------------------------
    
    files = [open(filename, "r") for filename in input_app_rp_filenames]

    with open(output_app_rp_filename, "w") as f_new:


        for f in files:
            assert f.readline().strip() == "core" 
        f_new.write("core\n")
        new_line = ' '.join(f.readline().strip() for f in files)
        f_new.write(f"{new_line}\n")

        for f in files:
            assert f.readline().strip() == "has_camera" 
        f_new.write("has_camera\n")
        new_line = ' '.join(f.readline().strip() for f in files)
        f_new.write(f"{new_line}\n")

        for f in files:
            assert f.readline().strip() == "has_gpu" 
        f_new.write("has_gpu\n")
        new_line = ' '.join(f.readline().strip() for f in files)
        f_new.write(f"{new_line}\n")

        for f in files:
            assert f.readline().strip() == "bandwidth" 
        f_new.write("bandwidth\n")
        for i in range(len(input_app_filenames)):
            for j in original_arcs[i]:
                line = files[i].readline().split()[1]
                f_new.write(f"{j} {line}\n")

        for f in files:
            assert f.readline().strip() == "latency" 
        f_new.write("latency\n")
        for i in range(len(input_app_filenames)):
            for j in original_arcs[i]:
                line = files[i].readline().split()[1]
                f_new.write(f"{j} {line}\n")

    for f in files:
        f.close()