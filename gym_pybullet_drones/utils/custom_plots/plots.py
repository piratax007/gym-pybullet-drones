#!/usr/bin/env python3

import csv
import os
import matplotlib.pyplot as plt


def get_data_from(file: str) -> tuple:
    data_file = os.path.join(GENERAL_PATH, file)

    with open(data_file, 'r') as f:
        data_set = list(csv.reader(f, delimiter=','))
        x = list(map(lambda row: float(row[0]), data_set))
        y = list(map(lambda row: float(row[1]), data_set))
        z = None if len(data_set[0]) < 3 else list(map(lambda row: float(row[2]), data_set))

    if z is not None:
        return x, y, z

    return x, y


def single_trace(**kwargs) -> None:
    _, axes = plt.subplots(1)

    x, y = get_data_from(kwargs['file'])

    axes.plot(x, y, kwargs['color'])
    axes.set_xlabel(kwargs['x_label'])
    axes.set_ylabel(kwargs['y_label'])
    axes.set_title(kwargs['title'])

    plt.show()
    
    
def multiple_traces(**kwargs) -> None:
    _, axes = plt.subplots(1)

    for i, file in enumerate(kwargs['files']):
        x, y = get_data_from(file)
        axes.plot(x, y, kwargs['colors'][i] if kwargs['colors'] != 'auto' else '')

    axes.set_xlabel(kwargs['x_label'])
    axes.set_ylabel(kwargs['y_label'])
    axes.set_title(kwargs['title'])

    plt.show()


def single_3D_trace(**kwargs) -> None:
    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')

    x, y, z = get_data_from(kwargs['file'])

    axes.plot(x, y, z, kwargs['color'] if kwargs['color'] != 'auto' else '')
    axes.set_xlabel(kwargs['x_label'])
    axes.set_ylabel(kwargs['y_label'])
    axes.set_zlabel(kwargs['z_label'])

    plt.show()


if __name__ == '__main__':
    GENERAL_PATH = os.path.dirname(
        '../../experiments/results/save-TEST-WITH-0.0052RPMS-SQUARED-DIFFERENCE-2M-03.19.2024_10.08.48/'
    )

    single_3D_trace(
        file='save-flight-starting-from-x0y1z0-03.19.2024_15.26.29/test_3D.csv',
        color='auto',
        x_label='x (m)',
        y_label='y (m)',
        z_label='z (m)',
        title='Flight Starting From [0 1 0]'
    )
