#!/usr/bin/env python3

import csv
import os
import matplotlib.pyplot as plt


def get_data_from_csv(file: str) -> tuple:
    data_file = os.path.join(GENERAL_PATH, file)

    with open(data_file, 'r') as f:
        data_set = tuple(csv.reader(f, delimiter=','))
        x = tuple(map(lambda row: float(row[0]), data_set))
        y = tuple(map(lambda row: float(row[1]), data_set))
        try:
            z = tuple(map(lambda row: float(row[2]), data_set))
            return x, y, z
        except IndexError:
            return x, y, None


def traces_from_csv(
        files: list,
        axis: plt.Axes,
        color_mode: str = 'auto',
        color_list: list = None
) -> None:
    for i, file in enumerate(files):
        data = get_data_from_csv(file)
        axis.plot(*data, color_list[i] if color_mode != 'auto' else '')


def traces_2D(**kwargs) -> None:
    # ToDo: DRY
    _, axis = plt.subplots(1)

    traces_from_csv(kwargs['files'], axis)

    if kwargs['axes_limits']['mode'] != 'auto':
        axis.set_xlim(*kwargs['axes_limits']['x_range'])
        axis.set_ylim(*kwargs['axes_limits']['y_range'])

    axis.set_xlabel(kwargs['x_label'])
    axis.set_ylabel(kwargs['y_label'])
    axis.set_title(kwargs['title'])

    plt.show()


def traces_3D(**kwargs) -> None:
    # ToDo: DRY
    font = {'family': 'serif', 'weight': 'bold', 'size': 15}
    plt.rc('font', **font)
    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')

    traces_from_csv(kwargs['files'], axis)

    if kwargs['axes_limits']['mode'] != 'auto':
        axis.set_xlim(*kwargs['axes_limits']['x_range'])
        axis.set_ylim(*kwargs['axes_limits']['y_range'])
        axis.set_zlim(*kwargs['axes_limits']['z_range'])

    axis.set_xlabel(kwargs['x_label'], labelpad=20)
    axis.set_ylabel(kwargs['y_label'], labelpad=20)
    axis.set_zlabel(kwargs['z_label'], labelpad=20)
    axis.set_title(kwargs['title'])

    plt.show()


if __name__ == '__main__':
    GENERAL_PATH = os.path.dirname(
        '../../experiments/results/save-TEST-WITH-0.0052RPMS-SQUARED-DIFFERENCE-2M-03.19.2024_10.08.48/'
    )

    traces_3D(
        files=['save-flight-starting-from-x0y1z0-03.19.2024_15.26.29/test_3D.csv'],
        colors='auto',
        x_label='x (m)',
        y_label='y (m)',
        z_label='z (m)',
        axes_limits=dict(
            mode='auto'
        ),
        title='Trajectory From [0 1 0]'
    )
