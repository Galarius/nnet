# -*- coding: utf-8 -*-
#!/usr/bin/env python3

__author__ = "Ilya Shoshin (Galarius)"

import os
import argparse
import numpy as np

from nnet import NeuralNetwork
from map import Map

MAP_WIDTH = 20
MAP_HEIGHT = 20
W_PREFIX = "weights"

np.set_printoptions(precision=2, suppress=True)


def main(argv):
    if args.seed:
        np.random.seed(args.seed)

    map = Map(MAP_WIDTH, MAP_HEIGHT)
    net = NeuralNetwork(2, args.layer_neurons, 1, args.hidden_layers, args.bias)
    print(net)
    if args.train:
        # Training data
        train_d0, train_d1 = map.dataset(0, MAP_WIDTH + MAP_HEIGHT), map.dataset(
            1, MAP_WIDTH + MAP_HEIGHT
        )
        td0 = np.array([[0]] * train_d0.shape[0], dtype=float)
        td1 = np.array([[1]] * train_d1.shape[0], dtype=float)
        t = np.concatenate((td0, td1), axis=0)  # Already normalized
        # Input
        x = np.concatenate((train_d0, train_d1), axis=0)
        x_normalized = x / np.amax(x, axis=0)

        print("Training...")
        if args.logging:
            with open("training.log", "w") as f:
                for epoch in range(args.epochs):
                    f.write(f"Epoch {epoch}\n")
                    f.write(f"Input:\n{x_normalized.T}\n")
                    f.write(f"Actual Output:\n{t.T}\n")
                    f.write(
                        f"Predicted Output:\n{np.round(net.forward(x_normalized).T)}\n"
                    )
                    f.write(
                        f"Loss:\n{str(np.mean(np.square(t - net.forward(x_normalized))))}\n\n"
                    )
                    net.train(x_normalized, t)
        else:
            for epoch in range(args.epochs):
                net.train(x_normalized, t, args.alpha, args.train_speed)
        print("Saving weights...")
        net.save_weights(W_PREFIX)
        print("Done.")
    else:
        train_d0 = train_d1 = np.array([])
        if os.path.exists("{}_0.w.txt".format(W_PREFIX)):
            print("Loading weights...")
            net.load_weights(W_PREFIX)
            print("Done.")
        else:
            print("No weights were found!")

    if args.seed:
        np.random.seed(args.seed + 1)

    # Input
    zds0, zds1 = np.random.randint(2, 20), np.random.randint(2, 20)
    d0, d1 = map.dataset(0, zds0), map.dataset(1, zds1)
    x = np.concatenate((d0, d1), axis=0)
    x_normalized = x / np.amax(x, axis=0)
    # Expected data to be tested
    td0 = np.array([[0]] * d0.shape[0], dtype=float)
    td1 = np.array([[1]] * d1.shape[0], dtype=float)
    t = np.concatenate((td0, td1), axis=0)  # Already normalized
    # Output
    y = np.round(net.predict(x_normalized))
    if args.verbose:
        print("Input:")
        print(x)
        print("Output (Expected):")
        print(t)
        print("Output (Actual):")
        print(y)

    res = y == t
    if res.all():
        print("\nAll Good!")
    else:
        print(f"{res.sum() * 100 / len(res)}% are good!")

    if args.plotting:
        # Filtering hits and misses
        good = []
        bad = []
        for i, v in enumerate(res):
            if v:
                good.append(x[i])
            else:
                bad.append(x[i])
        map.plot(np.array(good), np.array(bad), train_d0, train_d1, args.plot_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-b", "--bias", action="store_true", help="use bias neuron in hidden layer"
    )
    ap.add_argument(
        "-i", "--hidden-layers", type=int, default=1, help="number of hidden layers"
    )
    ap.add_argument(
        "-j",
        "--layer-neurons",
        type=int,
        default=3,
        help="number of neurons in hidden layers",
    )
    ap.add_argument("-t", "--train", action="store_true", help="perform training")
    ap.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1000,
        help="train with specified number of epochs",
    )
    ap.add_argument(
        "-a", "--alpha", type=float, default=1, help="gradient descent momentum"
    )
    ap.add_argument(
        "-x",
        "--train-speed",
        type=float,
        default=1,
        help="gradient descent train speed",
    )
    ap.add_argument("-s", "--seed", type=int, default=0, help="seed random generator")
    ap.add_argument(
        "-l",
        "--logging",
        action="store_true",
        help="write training process into training.log file",
    )
    ap.add_argument("-p", "--plotting", action="store_true", help="show plot")
    ap.add_argument("-n", "--plot-name", type=str, default="map.png", help="plot name")
    ap.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = ap.parse_args()
    main(args)
