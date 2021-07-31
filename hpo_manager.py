import itertools
from subprocess import Popen
import signal

ROIS = ['V1', 'V2', 'V3', 'V4', 'EBA', 'LOC', 'PPA', 'FFA', 'STS']
LAYERS = ['x1', 'x2', 'x3', 'x4']
COMB = list(itertools.product(ROIS, LAYERS))
ports = [29543+i for i in range(len(COMB))]
# python = '/home/huze/local_algo/bin/python'
python = '/home/huze/.conda/envs/algonauts/bin/python'

try:
    ps = []
    for port, (roi, layer) in zip(ports, COMB):
        p = Popen([python, "hpo.py", "--roi", roi, "--layer", layer, "--local_port", str(port)])
        ps.append(p)
    exit_codes = [p.wait() for p in ps]
    print(exit_codes)
except KeyboardInterrupt:
    for p in ps:
        p.send_signal(signal.SIGINT)
