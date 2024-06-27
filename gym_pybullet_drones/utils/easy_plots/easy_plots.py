import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PATH:
    def __init__(self, path):
        self.path = path


GENERAL_PATH = PATH("")


def _get_data_from_csv(file: str) -> tuple:
    data_file = os.path.join(GENERAL_PATH.path, file)

    with open(data_file, 'r') as f:
        data_set = tuple(csv.reader(f, delimiter=','))
        x = tuple(map(lambda row: float(row[0]), data_set))
        y = tuple(map(lambda row: float(row[1]), data_set))
        try:
            z = tuple(map(lambda row: float(row[2]), data_set))
            return x, y, z
        except IndexError:
            return x, y


def _export_tuple_to_csv(data: tuple, path: os.path, file_name: str) -> None:
    array_data = np.array(data)
    with open(path + file_name + ".csv", 'wb') as csv_file:
        np.savetxt(csv_file, np.transpose(array_data), delimiter=",")


def combine_data_from(files: list, **kwargs) -> tuple:
    combined_data = tuple(map(lambda file: _get_data_from_csv(file)[1], files))

    if kwargs['save_to_csv']:
        _export_tuple_to_csv(combined_data, kwargs['path'], kwargs['file_name'])

    return combined_data


def _plot_references(
        files: list,
        axis: plt.Axes,
        labeled: bool = True,
        label: str = 'Reference',
        style: str = '--'
) -> None:
    for i, file in enumerate(files):
        reference = _get_data_from_csv(file)
        if style == "point" and len(reference) == 3:
            axis.scatter(reference[0], reference[1], reference[2], color='black', label=label, s=100)
        else:
            axis.plot(
                *reference,
                color='black',
                linestyle=style,
                linewidth=1.5,
                label=label if (i == 0 and labeled) else ''
            )


def _interior_axes(create: bool, axes: plt.Axes, settings: dict):
    if create:
        interior_axes = axes.inset_axes(
            settings['x_y_width_height'],
            xlim=settings['x_portion'],
            ylim=settings['y_portion'],
        )
        interior_axes.set_xticklabels([])
        interior_axes.set_yticklabels([])

        return interior_axes

    return None


def _parse_references(references: dict) -> dict:
    if not references['show']:
        references['labeled'] = False
        references['label'] = ''
        references['files'] = []

    if references['show'] and not references['labeled']:
        references['label'] = ''

    if 'style' not in references.keys():
        references['style'] = '--'

    if not references['interior_detail']:
        references['interior_detail_settings'] = dict()

    return references


def _traces_from_csv(files: list, labels: list, axis: plt.Axes, references: dict, **colors: dict) -> None:
    parsed_references = _parse_references(references)

    interior_axes = _interior_axes(
        parsed_references['interior_detail'],
        axis,
        parsed_references['interior_detail_settings']
    )
    for i, file in enumerate(files):
        data = _get_data_from_csv(file)
        if interior_axes is not None:
            interior_axes.plot(*data, colors['color_list'][i] if colors['color_mode'] != 'auto' else '')
            axis.indicate_inset_zoom(interior_axes, edgecolor='black', alpha=0.25)
        axis.plot(*data, colors['color_list'][i] if colors['color_mode'] != 'auto' else '', label=labels[i])
        axis.legend()

    if parsed_references['show']:
        _plot_references(
            parsed_references['files'],
            axis,
            parsed_references['labeled'],
            parsed_references['label'],
            parsed_references['style']
        )



def _set_axis(axis: plt.Axes, **settings: dict) -> None:
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


def single_axis_2D(files: list, labels: list, references: dict, colors: dict, settings: dict) -> None:
    _, axis = plt.subplots(1)

    _traces_from_csv(files, labels, axis, references, **colors)
    _set_axis(axis, **settings)

    plt.show()


def single_axis_3D(files: list, labels: list, references: dict, colors: dict, settings: dict) -> None:
    plt.rcParams['text.usetex'] = True
    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')

    _traces_from_csv(
        files,
        labels,
        axis,
        references,
        **colors
    )
    _set_axis(axis, **settings)

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
            _traces_from_csv(
                content_specification[traces_key]['files'],
                content_specification[traces_key]['labels'],
                axis[row] if subplots['columns'] == 1 else axis[row, col],
                content_specification[traces_key]['references'],
                **colors
            )
            _set_axis(
                axis[row] if subplots['columns'] == 1 else axis[row, col],
                **content_specification[traces_key]['settings']
            )

    plt.show()


def animate(data: dict, references: dict, settings: dict, colors: dict, video_name: str = 'video') -> None:
    figure = plt.figure(figsize=(16, 9), dpi=720 / 16)
    axis = plt.gca()
    figure.subplots_adjust(left=0.13, right=0.87, top=0.85, bottom=0.15)
    _set_axis(axis, **settings)
    parsed_references = _parse_references(references)

    if parsed_references['show']:
        _plot_references(parsed_references['files'], axis)

    def update(frame_number):
        trace.set_xdata(x[:frame_number])
        trace.set_ydata(y[:frame_number])
        return trace

    for i, file in enumerate(data['files']):
        x, y = _get_data_from_csv(file)
        trace = axis.plot(x[0], y[0], colors['color_list'][i] if colors['color_mode'] != 'auto' else '')[0]
        trace.set_label(data['labels'][i])
        axis.legend()
        anim = animation.FuncAnimation(figure, update, frames=len(x), interval=3, repeat=False)
        anim.save(video_name + str(i) + '.mp4', 'ffmpeg', fps=30, dpi=300)


def animation_3D(data: dict, references: dict, settings: dict, color: str = 'red', video_name: str = 'video') -> None:
    figure = plt.figure(figsize=(16, 9), dpi=720 / 16)
    axis = figure.add_subplot(111, projection='3d')
    axis.view_init(elev=30, azim=45, roll=0)
    _set_axis(axis, **settings)

    x, y, z = _get_data_from_csv(data['files'][0])

    if references['show']:
        _traces_from_csv(
            references['files'],
            ['Reference'],
            axis,
            dict(show=False),
            **dict(color_mode='custom', color_list=['black'])
        )

    def update(frame_number):
        trace.set_data(x[:frame_number], y[:frame_number])
        trace.set_3d_properties(z[:frame_number])
        return axis

    trace, = axis.plot3D([], [], [], color)
    trace.set_label('Actual Trajectory')
    anim = animation.FuncAnimation(figure, update, frames=len(x), interval=3, repeat=False)
    anim.save(video_name + '.mp4', 'ffmpeg', fps=30, dpi=300)
