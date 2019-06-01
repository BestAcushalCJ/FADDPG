import numpy as np
import pyglet


pyglet.clock.set_fps_limit(10000)


# 环境及运动逻辑定义
class CarEnv(object):
    n_sensor = 5 # 距离传感器数量
    action_dim = 1
    state_dim = n_sensor #状态空间维度
    viewer = None
    viewer_xy = (600, 600) # 地图大小
    sensor_max = 60. # 传感器最大检测距离
    start_point = [40, 580] # 运动起始位置
    speed = 50.
    dt = 0.1


    def __init__(self, discrete_action=False):
        self.is_discrete_action = discrete_action
        if discrete_action:
            self.actions = [-1, 0, 1]  # 离散动作定义
        else:
            self.action_bound = [-1, 1] #连续动作定义范围

        self.terminal = False
        # node1 (x, y, r, w, l),中心坐标x、y，转过的角度r，尺寸w、l
        self.car_info = np.array([0, 0, 0, 20, 40], dtype=np.float64)   # 小车坐标

        # 四边形
        obstacle_coords = np.array([
            [170, 495],
            [255, 495],
            [255, 580],
            [170, 580],
        ])
        obstacle_coords1 = np.array([
            [360, 470],
            [490, 470],
            [490, 540],
            [360, 540],
        ])
        obstacle_coords2 = np.array([
            [35, 390],
            [85, 390],
            [85, 480],
            [35, 480],
        ])
        obstacle_coords3 = np.array([
            [350, 340],
            [530, 340],
            [530, 385],
            [350, 385],
        ])
        obstacle_coords4 = np.array([
            [120, 170],
            [190, 170],
            [190, 240],
            [120, 240],
        ])
        obstacle_coords5 = np.array([
            [375, 165],
            [460, 165],
            [460, 260],
            [375, 260],
        ])
        obstacle_coords6 = np.array([
            [20, 35],
            [130, 35],
            [130, 80],
            [20, 80],
        ])
        obstacle_coords7 = np.array([
            [430, 20],
            [550, 20],
            [550, 105],
            [430, 105],
        ])

        # 三角形
        obstacle_coords8 = np.array([
            [180, 315],
            [270, 315],
            [180, 410],
        ])
        obstacle_coords9 = np.array([
            [215, 50],
            [310, 80],
            [280, 160],
        ])
        self.all_obstacle_coords = np.array([obstacle_coords, obstacle_coords1, obstacle_coords2,
                                        obstacle_coords3, obstacle_coords4,obstacle_coords5,obstacle_coords6,
                                             obstacle_coords7, obstacle_coords8, obstacle_coords9])
        # 传感器参数：(distance, end_x, end_y) 5*3:每一行代表一个传感器
        self.sensor_info = self.sensor_max + np.zeros((self.n_sensor, 3))


    # each step:
    def step(self, action):
        # 获得当前动作
        if self.is_discrete_action:
            action = self.actions[action]
        else:
            action = np.clip(action, *self.action_bound)[0]

        self.car_info[2] += action * np.pi/30  # 一次最多max r = 6 degree
        # 看成一个近似的三角形处理，更新当前小车坐标
        self.car_info[:2] = self.car_info[:2] + \
                            self.speed * self.dt * np.array([np.cos(self.car_info[2]), np.sin(self.car_info[2])])

        # 判断当前的状态并返回
        self._update_sensor()
        s = self._get_state()
        # 获取即时奖励
        # 奖励规则：碰撞 -1； 无碰撞：0
        r = -1 if self.terminal else 0

        return s, r, self.terminal


    # 任务重置,在第一次开始迭代的时候也需要调用
    def reset(self):
        self.terminal = False
        self.car_info[:3] = np.array([*self.start_point, -np.pi/2])
        self._update_sensor()

        return self._get_state()


    # 环境渲染
    def render(self):
        if self.viewer is None:
            # View Definition: __init__(self, width, height, car_info, sensor_info, obstacle_coords)
            self.viewer = Viewer(*self.viewer_xy, self.car_info, self.sensor_info, self.all_obstacle_coords)
        self.viewer.render()


    # 产生一个随机动作,即转角
    def sample_action(self):
        if self.is_discrete_action:
            a = np.random.choice(list(range(3)))
        else:
            a = np.random.uniform(*self.action_bound, size=self.action_dim)
        return a


    # 设置当前刷新率
    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)


    # 获得当前是否探测到障碍物
    # 即状态空间
    def _get_state(self):
        s = self.sensor_info[:, 0].flatten()/self.sensor_max
        return s


    # 更新传感器的位置
    def _update_sensor(self):
        # 获得当前小车横纵坐标以及转角
        cx, cy, rotation = self.car_info[:3]
        n_sensors = len(self.sensor_info)
        # 获得五个传感器相对小车的角度（180度均匀生成）
        sensor_theta = np.linspace(-np.pi / 2, np.pi / 2, n_sensors)
        # 计算传感器在相对坐标系中末端点位置
        xs = cx + (np.zeros((n_sensors, ))+self.sensor_max) * np.cos(sensor_theta)
        ys = cy + (np.zeros((n_sensors, ))+self.sensor_max) * np.sin(sensor_theta)
        xys = np.array([[x, y] for x, y in zip(xs, ys)])    # shape (5 sensors, 2)
        # 获得末端点相对于坐标轴的差
        tmp_x = xys[:, 0] - cx
        tmp_y = xys[:, 1] - cy
        # apply rotation
        rotated_x = tmp_x * np.cos(rotation) - tmp_y * np.sin(rotation)
        rotated_y = tmp_x * np.sin(rotation) + tmp_y * np.cos(rotation)
        # 获得末端点在全局坐标系中的位置
        self.sensor_info[:, -2:] = np.vstack([rotated_x+cx, rotated_y+cy]).T

        q = np.array([cx, cy])
        # 对每个传感器做碰撞检测
        for si in range(len(self.sensor_info)):
            s = self.sensor_info[si, -2:] - q
            # 初始化
            possible_sensor_distance = [self.sensor_max]
            # 交叉点
            possible_intersections = [self.sensor_info[si, -2:]]

            # 障碍物碰撞检测
            current_obstacle_coords = self.all_obstacle_coords[0]
            for oi in range(len(current_obstacle_coords)):
                p = current_obstacle_coords[oi]
                # 一条边向量
                r = current_obstacle_coords[(oi + 1) % len(current_obstacle_coords)] - current_obstacle_coords[oi]
                if np.cross(r, s) != 0:  # 不平行，可能存在碰撞
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))

            current_obstacle_coords = self.all_obstacle_coords[1]
            for oi in range(len(current_obstacle_coords)):
                p = current_obstacle_coords[oi]
                # 一条边向量
                r = current_obstacle_coords[(oi + 1) % len(current_obstacle_coords)] - current_obstacle_coords[oi]
                if np.cross(r, s) != 0:  # 不平行，可能存在碰撞
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))

            current_obstacle_coords = self.all_obstacle_coords[2]
            for oi in range(len(current_obstacle_coords)):
                p = current_obstacle_coords[oi]
                # 一条边向量
                r = current_obstacle_coords[(oi + 1) % len(current_obstacle_coords)] - current_obstacle_coords[oi]
                if np.cross(r, s) != 0:  # 不平行，可能存在碰撞
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))

            current_obstacle_coords = self.all_obstacle_coords[3]
            for oi in range(len(current_obstacle_coords)):
                p = current_obstacle_coords[oi]
                # 一条边向量
                r = current_obstacle_coords[(oi + 1) % len(current_obstacle_coords)] - current_obstacle_coords[oi]
                if np.cross(r, s) != 0:  # 不平行，可能存在碰撞
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))

            # current_obstacle_coords = self.all_obstacle_coords[4]
            # for oi in range(len(current_obstacle_coords)):
            #     p = current_obstacle_coords[oi]
            #     # 一条边向量
            #     r = current_obstacle_coords[(oi + 1) % len(current_obstacle_coords)] - current_obstacle_coords[oi]
            #     if np.cross(r, s) != 0:  # 不平行，可能存在碰撞
            #         t = np.cross((q - p), s) / np.cross(r, s)
            #         u = np.cross((q - p), r) / np.cross(r, s)
            #         if 0 <= t <= 1 and 0 <= u <= 1:
            #             intersection = q + u * s
            #             possible_intersections.append(intersection)
            #             possible_sensor_distance.append(np.linalg.norm(u*s))
            #
            # current_obstacle_coords = self.all_obstacle_coords[5]
            # for oi in range(len(current_obstacle_coords)):
            #     p = current_obstacle_coords[oi]
            #     # 一条边向量
            #     r = current_obstacle_coords[(oi + 1) % len(current_obstacle_coords)] - current_obstacle_coords[oi]
            #     if np.cross(r, s) != 0:  # 不平行，可能存在碰撞
            #         t = np.cross((q - p), s) / np.cross(r, s)
            #         u = np.cross((q - p), r) / np.cross(r, s)
            #         if 0 <= t <= 1 and 0 <= u <= 1:
            #             intersection = q + u * s
            #             possible_intersections.append(intersection)
            #             possible_sensor_distance.append(np.linalg.norm(u*s))

            # 边界碰撞检测
            win_coord = np.array([
                [0, 0],
                [self.viewer_xy[0], 0],
                [*self.viewer_xy],
                [0, self.viewer_xy[1]],
                [0, 0],
            ])
            for oi in range(4):
                p = win_coord[oi]
                r = win_coord[(oi + 1) % len(win_coord)] - win_coord[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = p + t * r
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(intersection - q))

            # 获得传感器与边的最短距离
            distance = np.min(possible_sensor_distance)
            distance_index = np.argmin(possible_sensor_distance)
            self.sensor_info[si, 0] = distance
            self.sensor_info[si, -2:] = possible_intersections[distance_index]

            if distance < self.car_info[-1]/2:
                self.terminal = True


# 可视化定义
class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    # self.viewer = Viewer(*self.viewer_xy, self.car_info, self.sensor_info, self.obstacle_coords)
    def __init__(self, width, height, car_info, sensor_info, all_obstacle_coords):
        super(Viewer, self).__init__(width, height, resizable=False, caption='2D避障训练', vsync=False)
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.car_info = car_info
        self.sensor_info = sensor_info

        self.batch = pyglet.graphics.Batch()
        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)

        self.sensors = []
        # 一条直线由两个点组成，表示传感器的探测距离
        line_coord = [0, 0] * 2
        c = (73, 73, 73) * 2
        for i in range(len(self.sensor_info)):
            self.sensors.append(self.batch.add(2, pyglet.gl.GL_LINES, foreground, ('v2f', line_coord), ('c3B', c)))

        # 小车图形
        car_box = [0, 0] * 4
        c = (249, 86, 86) * 4
        self.car = self.batch.add(4, pyglet.gl.GL_QUADS, foreground, ('v2f', car_box), ('c3B', c))

        # 四边形障碍物
        c = (134, 181, 244) * 4
        self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', all_obstacle_coords[0].flatten()), ('c3B', c))
        self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', all_obstacle_coords[1].flatten()), ('c3B', c))
        self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', all_obstacle_coords[2].flatten()), ('c3B', c))
        self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', all_obstacle_coords[3].flatten()), ('c3B', c))
        self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', all_obstacle_coords[4].flatten()), ('c3B', c))
        self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', all_obstacle_coords[5].flatten()), ('c3B', c))
        self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', all_obstacle_coords[6].flatten()), ('c3B', c))
        self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', all_obstacle_coords[7].flatten()), ('c3B', c))
        # 三角形障碍物
        c = (134, 181, 244) * 3
        self.batch.add(3, pyglet.gl.GL_TRIANGLES, background, ('v2f', all_obstacle_coords[8].flatten()), ('c3B', c))
        self.batch.add(3, pyglet.gl.GL_TRIANGLES, background, ('v2f', all_obstacle_coords[9].flatten()), ('c3B', c))

    def render(self):
        pyglet.clock.tick()
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    # 在每一帧中需要变化的小车和传感器线
    def _update(self):
        cx, cy, r, w, l = self.car_info

        # sensors
        for i, sensor in enumerate(self.sensors):
            # 传感器起始点和末端点信息
            sensor.vertices = [cx, cy, *self.sensor_info[i, -2:]]

        # 小车端点信息
        xys = [
            [cx + l / 2, cy + w / 2],
            [cx - l / 2, cy + w / 2],
            [cx - l / 2, cy - w / 2],
            [cx + l / 2, cy - w / 2],
        ]
        r_xys = []
        for x, y in xys:
            tempX = x - cx
            tempY = y - cy
            # apply rotation
            rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
            rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
            # 计算在全局坐标系中的坐标
            x = rotatedX + cx
            y = rotatedY + cy
            r_xys += [x, y]
        self.car.vertices = r_xys


if __name__ == '__main__':
    np.random.seed(1)
    env = CarEnv()
    env.set_fps(1)
    for ep in range(30):
        s = env.reset()
        # for t in range(100):
        while True:
            env.render()
            s, r, done = env.step(env.sample_action())
            if done:
                break