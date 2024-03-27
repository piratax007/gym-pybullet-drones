#!/usr/bin/env python3

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from aquarel import load_theme


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
            return x, y


def plot_references(files: list, axis: plt.Axes) -> None:
    for i, file in enumerate(files):
        reference = get_data_from_csv(file)
        axis.plot(*reference, color='gray', linestyle='--', linewidth=1.5)


def traces_from_csv(files: list, axis: plt.Axes, references: dict, **colors: dict) -> None:
    for i, file in enumerate(files):
        data = get_data_from_csv(file)
        axis.plot(*data, colors['color_list'][i] if colors['color_mode'] != 'auto' else '')

    if references['view']:
        plot_references(references['files'], axis)


def set_axis(axis: plt.Axes, **settings: dict) -> None:
    # ToDo: Improve exceptions
    if settings['limits']['mode'] != 'auto':
        axis.set_xlim(*settings['limits']['x_range'])
        axis.set_ylim(*settings['limits']['y_range'])
        try:
            axis.set_zlim(*settings['limits']['z_range'])
        except:
            pass

    axis.set_xlabel(settings['labels']['x_label'], labelpad=20)
    axis.set_ylabel(settings['labels']['y_label'], labelpad=20)
    try:
        axis.set_zlabel(settings['labels']['z_label'], labelpad=20)
    except:
        pass

    axis.set_title(settings['labels']['title'])


def single_axis_2D(files: list, references: dict, colors: dict, settings: dict) -> None:
    _, axis = plt.subplots(1)

    traces_from_csv(files, axis, references, **colors)
    set_axis(axis, **settings)

    plt.show()


def single_axis_3D(files: list, colors: dict, settings: dict) -> None:
    font = {'family': 'serif', 'weight': 'bold', 'size': 15}
    plt.rc('font', **font)
    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')

    traces_from_csv(files, axis, dict(view=False), **colors)
    set_axis(axis, **settings)

    plt.show()


def multiple_axis_2D(
        subplots: dict,
        content_specification: dict,
        colors: dict
) -> None:
    _, axis = plt.subplots(subplots['rows'], subplots['columns'])

    for col in range(subplots['columns']):
        for row in range(subplots['rows']):
            traces_key = str(f"({row}, {col})")
            traces_from_csv(
                content_specification[traces_key]['files'],
                axis[row] if subplots['columns'] == 1 else axis[row, col],
                content_specification[traces_key]['references'],
                **colors
            )
            set_axis(
                axis[row] if subplots['columns'] == 1 else axis[row, col],
                **content_specification[traces_key]['settings']
            )

    plt.show()


def animate(files: list, references: dict, settings: dict, colors: dict, video_name: str = 'video') -> None:
    figure = plt.figure(figsize=(16, 9), dpi=720 / 16)
    axis = plt.gca()
    set_axis(axis, **settings)

    if references['view']:
        plot_references(references['files'], axis)

    def update(frame_number):
        trace.set_xdata(x[:frame_number])
        trace.set_ydata(y[:frame_number])
        return trace

    for i, file in enumerate(files):
        x, y = get_data_from_csv(file)
        trace = axis.plot(x[0], y[0], colors['color_list'][i] if colors['color_mode'] != 'auto' else '')[0]
        anim = animation.FuncAnimation(figure, update, frames=len(x), interval=3, repeat=False)
        anim.save(video_name + str(i) + '.mp4', 'ffmpeg', fps=30, dpi=300)


if __name__ == '__main__':
    GENERAL_PATH = os.path.dirname(
        '../../experiments/results/save-TEST-WITH-0.0052RPMS-SQUARED-DIFFERENCE-2M-03.19.2024_10.08.48/'
    )
    theme = (
        load_theme('scientific')
        .set_font(family='serif', size=20)
        .set_title(pad=20)
        .set_axes(bottom=True, top=True, left=True, right=True, xmargin=0, ymargin=0, zmargin=0, width=2)
        .set_grid(style='--', width=1)
        .set_ticks(draw_minor=True, pad_major=10)
        .set_lines(width=2.5)
    )
    theme.apply()

    animate(files=[
        'save-flight-starting-from-x0y1z0-03.19.2024_15.26.29/x0.csv',
        'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/x0.csv'
    ],
        references=dict(
            view=True,
            files=[
                'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/x_low_ref.csv',
                'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/x_high_ref.csv',
            ]
        ),
        settings=dict(
            limits=dict(
                mode='custom',
                x_range=(0, 15),
                y_range=(-1, 1)
            ),
            labels=dict(
                x_label='t (s)',
                y_label='y (m)',
                title="Flight starting from random x position"
            )
        ),
        colors=dict(
            color_mode='custom',
            color_list=['red', 'green']
        )
    )

    # single_axis_2D(
    #     files=[
    #         'save-flight-starting-from-x0y1z0-03.19.2024_15.26.29/x0.csv',
    #         'save-flight-starting-from-x0y1z2-03.19.2024_15.52.13/x0.csv'
    #     ],
    #     references=dict(view=False),
    #     colors=dict(
    #         color_mode='custom',
    #         color_list=['red', 'green']
    #     ),
    #     settings=dict(
    #         limits=dict(mode='auto'),
    #         labels=dict(
    #             x_label='t (s)',
    #             y_label='x (m)',
    #             title='Multiple traces using the same axis'
    #         )
    #     )
    # )

    # single_axis_3D(
    #     files=[
    #         'save-flight-starting-from-x0y1z0-03.19.2024_15.26.29/test_3D.csv',
    #         'save-flight-starting-from-x1ym1z2-03.26.2024_16.52.52/test_3D.csv',
    #         'save-flight-starting-from-xm05ym075z0-03.26.2024_17.27.35/test_3D.csv'
    #     ],
    #     colors=dict(color_mode='auto'),
    #     settings=dict(
    #         limits=dict(
    #             mode='custom',
    #             x_range=(-1, 1),
    #             y_range=(-1, 1),
    #             z_range=(0, 2)
    #         ),
    #         labels=dict(
    #             x_label='x (m)',
    #             y_label='y (m)',
    #             z_label='z (m)',
    #             title="Trajectory From [0 1 0]"
    #         )
    #     )
    # )

    # multiple_axis_2D(
    #     subplots=dict(rows=3, columns=1),
    #     content_specification={
    #         '(0, 0)': dict(
    #             files=[
    #                 'save-flight-starting-from-x0y1z0-03.19.2024_15.26.29/x0.csv',
    #                 'save-flight-starting-from-x0y1z2-03.19.2024_15.52.13/x0.csv',
    #                 'save-flight-starting-from-x1y0z2-03.19.2024_15.41.02/x0.csv',
    #                 'save-flight-starting-from-xm05y1z025-03.27.2024_10.19.39/x0.csv',
    #                 'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/x0.csv'
    #             ],
    #             references=dict(
    #                 view=True,
    #                 files=[
    #                     'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/x_low_ref.csv',
    #                     'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/x_high_ref.csv',
    #                 ]
    #             ),
    #             settings=dict(
    #                 limits=dict(
    #                     mode='custom',
    #                     x_range=(0, 15),
    #                     y_range=(-1, 1)
    #                 ),
    #                 labels=dict(
    #                     x_label='t (s)',
    #                     y_label='x (m)',
    #                     title=''
    #                 )
    #             )),
    #         '(1, 0)': dict(
    #             files=[
    #                 'save-flight-starting-from-x0y1z0-03.19.2024_15.26.29/y0.csv',
    #                 'save-flight-starting-from-x0y1z2-03.19.2024_15.52.13/y0.csv',
    #                 'save-flight-starting-from-x1y0z2-03.19.2024_15.41.02/y0.csv',
    #                 'save-flight-starting-from-xm05y1z025-03.27.2024_10.19.39/y0.csv',
    #                 'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/y0.csv'
    #             ],
    #             references=dict(
    #                 view=True,
    #                 files=[
    #                     'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/y_low_ref.csv',
    #                     'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/y_high_ref.csv',
    #                 ]
    #             ),
    #             settings=dict(
    #                 limits=dict(
    #                     mode='custom',
    #                     x_range=(0, 15),
    #                     y_range=(-1, 1)
    #                 ),
    #                 labels=dict(
    #                     x_label='t (s)',
    #                     y_label='y (m)',
    #                     title=''
    #                 )
    #             )),
    #         '(2, 0)': dict(
    #             files=[
    #                 'save-flight-starting-from-x0y1z0-03.19.2024_15.26.29/z0.csv',
    #                 'save-flight-starting-from-x0y1z2-03.19.2024_15.52.13/z0.csv',
    #                 'save-flight-starting-from-x1y0z2-03.19.2024_15.41.02/z0.csv',
    #                 'save-flight-starting-from-xm05y1z025-03.27.2024_10.19.39/z0.csv',
    #                 'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/z0.csv'
    #             ],
    #             references=dict(
    #                 view=True,
    #                 files=[
    #                     'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/z_low_ref.csv',
    #                     'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/z_high_ref.csv',
    #                 ]
    #             ),
    #             settings=dict(
    #                 limits=dict(
    #                     mode='custom',
    #                     x_range=(0, 15),
    #                     y_range=(0, 2)
    #                 ),
    #                 labels=dict(
    #                     x_label='t (s)',
    #                     y_label='z (m)',
    #                     title=''
    #                 )
    #             )),
    #     },
    #     colors=dict(color_mode='auto'),
    # )
