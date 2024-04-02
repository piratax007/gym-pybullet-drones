import csv
import os
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


def _plot_references(files: list, axis: plt.Axes) -> None:
    for i, file in enumerate(files):
        reference = _get_data_from_csv(file)
        axis.plot(*reference, color='gray', linestyle='--', linewidth=1.5, label='Reference' if i == 0 else '')
        axis.legend()


def _traces_from_csv(files: list, labels:list, axis: plt.Axes, references: dict, **colors: dict) -> None:
    for i, file in enumerate(files):
        data = _get_data_from_csv(file)
        axis.plot(*data, colors['color_list'][i] if colors['color_mode'] != 'auto' else '', label=labels[i])
        axis.legend()

    if references['show']:
        _plot_references(references['files'], axis)


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


def single_axis_3D(files: list, labels: list, colors: dict, settings: dict) -> None:
    font = {'family': 'serif', 'weight': 'bold', 'size': 15}
    plt.rc('font', **font)
    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')

    _traces_from_csv(files, labels, axis, dict(show=False), **colors)
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
    _set_axis(axis, **settings)

    if references['view']:
        _plot_references(references['files'], axis)

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
