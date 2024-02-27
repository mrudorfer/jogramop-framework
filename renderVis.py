import sys, glob, os


fout = open("all-vis.sh", "wt")

for sdir in glob.glob("results/*"):
    scenario = os.path.split(sdir)[1]
    print(sdir, "-", scenario)
    scenario = int(scenario)
    for file in glob.glob("{}/*/*00.try".format(sdir)):
        prefix = file[:-4]
        out = "{}-000.png".format(prefix)
        cmd = "python3.10 visualizationTrajs.py {} {} {} ".format(scenario, file, prefix)
        if os.path.isfile(out):
            cmd = "#" + cmd

        fout.write(cmd + "\n")

fout.close()        
