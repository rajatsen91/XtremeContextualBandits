import argparse
import os
import numpy as np
from pathlib import Path
import scipy.sparse as smat
from sklearn.preprocessing import normalize


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        metavar="PATH",
        help="path to input text file",
    )

    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        required=True,
        metavar="DIR",
        help="path to save sparse matrices at.",
    )

    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
    )
    return parser


def convert(args):
    i = 0
    xdata = []
    xi = []
    xj = []
    ydata = []
    yi = []
    yj = []
    with open(args.input_file, "r") as fp:
        for line in fp:
            line = line.strip()
            if i == 0:
                line = line.split()
                n = int(line[0])
                p = int(line[1])
                l = int(line[2])
                i += 1
                continue
            split_line = line.split(" ")
            labels = split_line[0].split(",")
            features = split_line[1::]
            try:
                yj += [int(label) for label in labels]
                ydata += [1] * len(labels)
                yi += [i - 1] * len(labels)
                for f in features:
                    xi.append(i - 1)
                    fplit = f.split(":")
                    xj.append(int(fplit[0]))
                    if xj[-1] > p:
                        p = xj[-1]
                    xdata.append(float(fplit[1]))
                i += 1
            except:
                continue

    os.makedirs(args.output_folder, exist_ok=True)
    xmatrix = smat.coo_matrix((xdata, (xi, xj)), shape=(i - 1, p), dtype=np.float32).tocsr()
    ymatrix = smat.coo_matrix((ydata, (yi, yj)), shape=(i - 1, l), dtype=np.int32).tocsr()
    if args.normalize:
        xmatrix = normalize(xmatrix, norm="l2")
    smat.save_npz(Path(args.output_folder, "X.npz"), xmatrix)
    smat.save_npz(Path(args.output_folder, "Y.npz"), ymatrix)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    convert(args)
