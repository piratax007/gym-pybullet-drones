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


def combine_data_from(files: list, save_to_csv: bool = False, path: str = '', file_name: str = '') -> tuple:
    combined_data = tuple(map(lambda file: _get_data_from_csv(file)[1], files))

    if save_to_csv:
        _export_tuple_to_csv(combined_data, path, file_name)

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


def _interior_axes(create: bool, axes: plt.Axes, settings: dict) -> any:
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
            axis.indicate_inset_zoom(interior_axes, edgecolor='gray', alpha=0.25)
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

    # axis.legend()
    axis.legend(bbox_to_anchor=(0, 1, 1, 0.75), loc="lower left", borderaxespad=0, ncol=4)


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
    # axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axis.set_aspect('equal')


def _add_vertical_lines(axes: plt.Axes, x_positions: list, y_min: float = 0, y_max: float = 1, label: str = '') -> None:
    for i in range(len(x_positions)):
        axes.axvline(x_positions[i], y_min, y_max, ls='-.', color='gray', label=label if (i == 0) else '')

    # axes.legend()


def single_axis_2D(files: list, labels: list, references: dict, colors: dict, settings: dict) -> None:
    _, axis = plt.subplots(1)

    _traces_from_csv(files, labels, axis, references, **colors)
    _set_axis(axis, **settings)

    # _add_vertical_lines(axis, x_positions=[6000000, 6200000, 7000000], y_min=0.0, y_max=1.0, label='Training stopped')
    # _add_vertical_lines(axis, x_positions=[17950000], y_min=0.0, y_max=0.5)

    plt.show()


def _select_equally_spaced_sample(data: tuple, sample_size: int) -> tuple:
    step = len(data[0]) // sample_size
    r = tuple(tuple(inner_tuple[i * step] for i in range(sample_size)) for inner_tuple in data)
    return r


def single_axis_3D(
        files: list,
        labels: list,
        references: dict,
        colors: dict,
        settings: dict,
        decorations: dict = None,
) -> None:
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

    if decorations['show']:
        for i in range(len(decorations['position_files'])):
            positions = _select_equally_spaced_sample(
                combine_data_from(decorations['position_files'][i]),
                decorations['samples']
            )
            euler_angles = _select_equally_spaced_sample(
                combine_data_from(decorations['euler_angles_files'][i]),
                decorations['samples']
            )
            add_body_frame(positions, euler_angles, axis)

    # WAY POINT TRACKER CYLINDERS
    # add_cylinder(axis, center=(0, 1, 0.5))
    # add_cylinder(axis, center=(-1, 0, 1))
    # add_cylinder(axis, center=(-2, 1, 1.25))
    # add_cylinder(axis, center=(-3, 0, 1.5))

    # REWARD FUNCTION DIAGRAM
    add_cylinder(axis, center=(0, 0, 1))
    add_cylinder(axis, radius=2.1, center=(0, 0, 1), color='black')
    add_sphere(axis)
    add_double_arrow_annotation(axis, start=(-0.6489, -1.9971, 2), end=(-0.618, -1.902, 2), label='$\delta_R$')
    add_double_arrow_annotation(axis, start=(0, 0, 1), end=(0, 0, 2), label='$\delta_H$')
    add_double_arrow_annotation(axis, start=(0.24, -0.35, 0.36), end=(0, 0, 1), label='$T_e$')
    add_double_arrow_annotation(axis, start=(0, 0, 1), end=(0.058, 0.080, 1.030), label='$\Delta_p$', label_position='end')

    _set_axis(axis, **settings)

    plt.show()


def multiple_axis_2D(
        subplots: dict,
        content_specification: dict,
        colors: dict
) -> None:
    plt.rcParams['text.usetex'] = True
    fig, axis = plt.subplots(subplots['rows'], subplots['columns'])
    fig.align_labels()

    for col in range(subplots['columns']):
        for row in range(subplots['rows']):
            traces_key = str(f"({row}, {col})")
            # _add_vertical_lines(axis[row], x_positions=[8.3, 18.4, 29.5], label='Disturbances' if row == 0 else '')
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


def _euler_to_rotation_matrix(euler_angles: tuple) -> np.ndarray:
    angles_in_radians= tuple(np.deg2rad(euler_angles))
    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(angles_in_radians[0]), -np.sin(angles_in_radians[0])],
        [0, np.sin(angles_in_radians[0]), np.cos(angles_in_radians[0])]
    ])

    rotation_y = np.array([
        [np.cos(angles_in_radians[1]), 0, np.sin(angles_in_radians[1])],
        [0, 1, 0],
        [-np.sin(angles_in_radians[1]), 0, np.cos(angles_in_radians[1])]
    ])

    rotation_z = np.array([
        [np.cos(angles_in_radians[2]), -np.sin(angles_in_radians[2]), 0],
        [np.sin(angles_in_radians[2]), np.cos(angles_in_radians[2]), 0],
        [0, 0, 1]
    ])

    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
    return rotation_matrix


def add_body_frame(positions: tuple, attitudes: tuple, axes: plt.Axes) -> None:
    def arrange(data: tuple):
        arranged_data = []
        for row in range(len(data[0])):
            arranged_data.append((data[0][row], data[1][row], data[2][row]))

        return tuple(arranged_data)

    rotation_matrix = tuple(map(lambda a: _euler_to_rotation_matrix(a), arrange(attitudes)))
    arranged_positions = arrange(positions)

    for i in range(len(arranged_positions)):
        origin = np.array(arranged_positions[i])
        body_frame_x = rotation_matrix[i] @ np.array([1, 0, 0])
        body_frame_y = rotation_matrix[i] @ np.array([0, 1, 0])
        body_frame_z = rotation_matrix[i] @ np.array([0, 0, 1])

        axes.quiver(*origin, *body_frame_x, color='green', length=0.2, normalize=True)
        axes.quiver(*origin, *body_frame_y, color='red', length=0.2, normalize=True)
        axes.quiver(*origin, *body_frame_z, color='blue', length=0.2, normalize=True)


def add_cylinder(
        axes: plt.Axes,
        radius: float = 2.0,
        height: float = 2.0,
        center: tuple = (0, 0, 1),
        color: str = 'blue'
) -> None:
    z = np.linspace(0, height, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(theta_grid) + center[1]
    z_grid = z_grid + center[2] - height / 2

    axes.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, rstride=5, cstride=5, color=color)


def add_sphere(axes: plt.Axes, center: tuple = (0, 0, 0.978), radius: float = 0.1) -> None:
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    x_grid = radius * np.sin(phi_grid) * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(phi_grid) * np.sin(theta_grid) + center[1]
    z_grid = radius * np.cos(phi_grid) + center[2]

    axes.plot_surface(x_grid, y_grid, z_grid, alpha=0.25, rstride=5, cstride=5, color='red')


def add_double_arrow_annotation(
        axes: plt.Axes,
        start: tuple,
        end: tuple,
        label: str = '',
        label_position: str = 'mid',
        color: str = 'black'
) -> None:
    axes.quiver(
        start[0], start[1], start[2],
        end[0] - start[0], end[1] - start[1], end[2] - start[2],
        color=color, arrow_length_ratio=0.1, linewidth=1
    )

    axes.quiver(
        end[0], end[1], end[2],
        start[0] - end[0], start[1] - end[1], start[2] - end[2],
        color=color, arrow_length_ratio=0.1, linewidth=1.25
    )

    if label_position == 'mid':
        mid_point = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2)
        axes.text(mid_point[0], mid_point[1], mid_point[2], label, color=color, fontsize=28)
    elif label_position == 'start':
        axes.text(start[0], start[1], start[2], label, color=color, fontsize=28)
    elif label_position == 'end':
        axes.text(end[0], end[1], end[2], label, color=color, fontsize=28)
