import random
import numpy as np
from scipy.interpolate import splev, splprep
import matplotlib.pyplot as plt


def gen_traj(rng, dynamic_paras: dict, traj_mode="line", sim_freq=240, control_freq=20, max_steps=200):
    """
    generate trajectories in polar coordinate system.
    sim_freq: scene simulation frequency
    dynamic_paras:
        speed: the speed of the conveyor, m/s
        dist: root & traj_line; radius of circle_line
        angle: forward & root-footpoint
        rotz_inv: move direction (True for CCW, False for CW)
        vary_speed: True(constant speed), False(random vel < speed).
    traj_mode:
        line: move along a line
        circle: move along a circle
        random: move within a square. dist: xy range(+-dist), angle: x offset
        random3d: move within a box. dist xyz range(+-dist), angle/length: x/z offset
    """
    speed, dist, angle, length, rotz_inv, vary_speed = dynamic_paras.pop('speed', 0.05), \
                                                       dynamic_paras.pop('dist', None), \
                                                       dynamic_paras.pop('angle', None), \
                                                       dynamic_paras.pop('length', None), \
                                                       dynamic_paras.pop('rotz_inv', True), \
                                                       dynamic_paras.pop('vary_speed', False)
    length = speed * max_steps / control_freq + 0.01 if length is None else float(length)
    dist = rng.uniform(0.3, 0.7) if dist is None else float(dist)
    angle = rng.uniform(-np.pi/2, np.pi/2) if angle is None else float(angle)
    rotz_inv = rng.choice([True, False], 1)[0] if rotz_inv is None else bool(rotz_inv)
    direct_para = -1 if rotz_inv else 1
    traj_init_ratio = 1. / 4
    num_steps = max_steps * sim_freq // control_freq
    if traj_mode == "random2d":
        dist = 0.56 if dist is None else float(dist)  # x offset
        angle = rng.uniform(0.15, 0.25) if angle is None else float(angle)
        xy_range = np.array([
            [dist - angle, dist + angle],
            [-angle, angle]
        ])
        random_xy = rng.uniform(xy_range[:, 0], xy_range[:, 1])
        traj_xy = np.tile(random_xy, (num_steps, 1))
        traj_xyz = np.concatenate((traj_xy, np.zeros((num_steps, 1))), axis=1)
        return traj_xyz
    num_steps = int(length * sim_freq / speed)
    n_segments = 10 if vary_speed else 0
    if traj_mode == "line":
        center_xy = np.array([dist * np.cos(angle), -dist * np.sin(angle)])
        start_xy = center_xy + direct_para * np.array([length * np.sin(angle), length * np.cos(angle)]) * traj_init_ratio
        end_xy = center_xy - direct_para * np.array([length * np.sin(angle), length * np.cos(angle)]) * (1 - traj_init_ratio)
        if not vary_speed:
            traj_xy = np.linspace(start_xy, end_xy, num_steps)  # (N, 2)
        else:
            ## continual bezier variable speed
            speeds_seg = rng.uniform(0.4, 1.2, n_segments) * speed
            speeds_pts = np.stack((np.arange(n_segments), speeds_seg)).T
            speeds = gen_bezier(speeds_pts, num_steps)
            line_pos = np.cumsum(speeds[:, 1] / sim_freq)
            traj_xy = start_xy - direct_para * np.array([line_pos * np.sin(angle), line_pos * np.cos(angle)]).T
        traj_xyz = np.concatenate((traj_xy, np.zeros((num_steps, 1))), axis=1)
    elif traj_mode == "circle":
        start_angle, end_angle = angle + direct_para * length * traj_init_ratio / dist, angle - direct_para * length * (1 - traj_init_ratio) / dist
        if not vary_speed:
            theta_steps = np.linspace(start_angle, end_angle, num_steps)
            traj_xy = np.array([dist * np.cos(theta_steps), -dist * np.sin(theta_steps)]).T  # (N, 2)
        else:
            ## continual bezier variable speed
            speeds_seg = rng.uniform(0.5, 1.5, n_segments) * speed
            speeds_pts = np.stack((np.arange(n_segments), speeds_seg)).T
            speeds = gen_bezier(speeds_pts, num_steps)
            line_angles = start_angle - direct_para * np.cumsum(speeds[:, 1] / sim_freq)
            traj_xy = np.array([dist * np.cos(line_angles), -dist * np.sin(line_angles)]).T
        traj_xyz = np.concatenate((traj_xy, np.zeros((num_steps, 1))), axis=1)
    elif traj_mode == "circular":
        y_offset = rng.uniform(-0.1, 0.1)
        center_xy = np.array([dist, y_offset])
        start_q = rng.uniform(0, 2 * np.pi)
        if not vary_speed:
            theta_steps = np.linspace(start_q, length, num_steps)
            traj_xy = center_xy + direct_para * angle * np.array([np.cos(theta_steps), -np.sin(theta_steps)]).T  # (N, 2)
        else:
            ## continual bezier variable speed
            speeds_seg = rng.uniform(0.5, 1.5, n_segments) * speed
            speeds_pts = np.stack((np.arange(n_segments), speeds_seg)).T
            speeds = gen_bezier(speeds_pts, num_steps)
            line_angles = start_q + direct_para * np.cumsum(speeds[:, 1] / sim_freq)
            traj_xy = center_xy + direct_para * angle * np.array([np.cos(line_angles), -np.sin(line_angles)]).T
        traj_xyz = np.concatenate((traj_xy, np.zeros((num_steps, 1))), axis=1)
    elif traj_mode == "bezier2d":
        dist = 0.56 if dist is None else float(dist)
        angle = rng.uniform(0.08, 0.12) if angle is None else float(angle)
        x_min, x_max, y_min, y_max = dist - angle, dist + angle, -angle, angle
        num_points = rng.randint(4, 9)
        initial_interval = 0.01
        final_interval = speed / sim_freq
        x = rng.uniform(x_min, x_max, num_points)
        y = rng.uniform(y_min, y_max, num_points)
        # Fit a B-spline to these control points
        tck, _ = splprep([x, y], s=0)
        x_100_samples, y_100_samples = generate_equal_distance_points_vectorized(tck, initial_interval, length)
        x_samples, y_samples = linear_interpolate_sequence(x_100_samples, y_100_samples, final_interval)
        x_adjusted = np.clip(x_samples, x_min, x_max)
        y_adjusted = np.clip(y_samples, y_min, y_max)
        xy_s = np.concatenate((x_adjusted[:, None], y_adjusted[:, None]), axis=1)
        cur_steps = xy_s.shape[0]
        if cur_steps < num_steps:
            traj_xy = np.concatenate((xy_s, xy_s[-1, None].repeat(num_steps - cur_steps, axis=0)), axis=0)
        else:
            traj_xy = xy_s[:num_steps]
        traj_xyz = np.concatenate((traj_xy, np.zeros((num_steps, 1))), axis=1)
    else:
        raise NotImplementedError

    return traj_xyz


def generate_equal_distance_points_vectorized(tck, interval, target_length, precision=1e-4):
    # 预先计算
    u_fine = np.linspace(0, 1, 10000)
    x_fine, y_fine = splev(u_fine, tck)
    dx_fine = np.diff(x_fine)
    dy_fine = np.diff(y_fine)
    distances_fine = np.sqrt(dx_fine**2 + dy_fine**2)
    cum_distances_fine = np.insert(np.cumsum(distances_fine), 0, 0)

    def find_next_point_vectorized(current_u_index, d):
        target_distance = cum_distances_fine[current_u_index] + d
        next_u_index = np.searchsorted(cum_distances_fine, target_distance)
        return next_u_index

    u_indices = [0]
    while cum_distances_fine[u_indices[-1]] < target_length:
        next_u_index = find_next_point_vectorized(u_indices[-1], interval)
        if next_u_index >= len(u_fine) or cum_distances_fine[next_u_index] > target_length:
            break
        u_indices.append(next_u_index)

    u_samples = u_fine[u_indices]
    x_samples, y_samples = splev(u_samples, tck)
    return x_samples, y_samples


def linear_interpolate_sequence(x_samples, y_samples, final_interval):
    interpolated_x = []
    interpolated_y = []

    for i in range(len(x_samples) - 1):
        interpolated_x.append(x_samples[i])
        interpolated_y.append(y_samples[i])

        distance = np.sqrt((x_samples[i+1] - x_samples[i])**2 + (y_samples[i+1] - y_samples[i])**2)
        num_inserts = int(distance / final_interval) - 1

        for j in range(1, num_inserts + 1):
            t = j / (num_inserts + 1)
            interpolated_x.append(x_samples[i] + t * (x_samples[i+1] - x_samples[i]))
            interpolated_y.append(y_samples[i] + t * (y_samples[i+1] - y_samples[i]))

    interpolated_x.append(x_samples[-1])
    interpolated_y.append(y_samples[-1])

    return np.array(interpolated_x), np.array(interpolated_y)


def gen_bezier(points: np.ndarray, num_points=2400, order=2):
    if order == 2:
        bezier = lambda t, p0, p1, p2: (1 - t)**2 * p0 + 2 * t * (1 - t) * p1 + t**2 * p2
    else:
        raise NotImplementedError
    pts_num = points.shape[0]
    assert pts_num > 2
    seg_pts_num = int(num_points / (pts_num - 2)) + 1
    central_points = (points[0:-1] + points[1:])/2  # N - 1
    bezier_pts = []
    for pid in range(pts_num-2):
        bezier_pts.extend(np.array(
            [bezier(t, central_points[pid], points[pid+1], central_points[pid+1]) for t in np.linspace(0, 1, seg_pts_num)]
        ))
    curve = np.array(bezier_pts)[:num_points]
    return curve


if __name__ == '__main__':
    ## generate random speeds aligning a bezier curve (10 segments)
    # n_seg = 10
    # vels = np.random.uniform(0.2, 1.0, 10)
    # points_seq = np.stack((np.arange(n_seg), vels)).T
    # bezier_curve = gen_bezier(points_seq)
    # print(bezier_curve.shape)
    # import matplotlib.pyplot as plt
    # plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'b-')
    # plt.plot(points_seq[:, 0], points_seq[:, 1], 'r.')
    # plt.show()

    ## generate random 2D bezier trajectory
    n_seg = random.randint(3, 10)
    x_min, x_max, y_min, y_max = 0.5, 1.0, -0.3, 0.3
    x_seg = np.random.uniform(x_min, x_max, n_seg)
    y_seg = np.random.uniform(y_min, y_max, n_seg)
    x_seg.sort()
    # y_seg.sort()
    points_seq = np.stack((x_seg, y_seg)).T
    print(points_seq.shape)
    bezier_curve = gen_bezier(points_seq)
    print(bezier_curve.shape)
    import matplotlib.pyplot as plt
    plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'b-')
    plt.plot(points_seq[:, 0], points_seq[:, 1], 'r.')
    plt.show()
