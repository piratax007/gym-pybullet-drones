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


def traces_from_csv(files: list, axis: plt.Axes, **colors: dict) -> None:
    for i, file in enumerate(files):
        data = get_data_from_csv(file)
        axis.plot(*data, colors['color_list'][i] if colors['color_mode'] != 'auto' else '')


def set_axis(axis: plt.Axes, **settings: dict) -> None:
    if settings['limits']['mode'] != 'auto':
        axis.set_xlim(*settings['limits']['x_range'])
        axis.set_ylim(*settings['limits']['y_range'])
        try:
            axis.set_zlim(*settings['limits']['z_range'])
        except KeyError:
            pass

    axis.set_xlabel(settings['labels']['x_label'], labelpad=20)
    axis.set_ylabel(settings['labels']['y_label'], labelpad=20)
    try:
        axis.set_zlabel(settings['labels']['z_label'], labelpad=20)
    except KeyError:
        pass

    axis.set_title(settings['labels']['title'])


def single_axis_2D(files: list, colors: dict, settings: dict) -> None:
    _, axis = plt.subplots(1)

    traces_from_csv(files, axis, **colors)
    set_axis(axis, **settings)

    plt.show()


def single_axis_3D(files: list, colors: dict, settings: dict) -> None:
    font = {'family': 'serif', 'weight': 'bold', 'size': 15}
    plt.rc('font', **font)
    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')

    traces_from_csv(files, axis, **colors)
    set_axis(axis, **settings)

    plt.show()


if __name__ == '__main__':
    GENERAL_PATH = os.path.dirname(
        '../../experiments/results/save-TEST-WITH-0.0052RPMS-SQUARED-DIFFERENCE-2M-03.19.2024_10.08.48/'
    )

    single_axis_3D(
        files=['save-flight-starting-from-x0y1z0-03.19.2024_15.26.29/test_3D.csv'],
        colors=dict(color_mode='auto'),
        settings=dict(
            limits=dict(
                mode='custom',
                x_range=(-1, 1),
                y_range=(-1, 1),
                z_range=(0, 2)
            ),
            labels=dict(
                x_label='x (m)',
                y_label='y (m)',
                z_label='z (m)',
                title="Trajectory From [0 1 0]"
            )
        ),
    )
