import numpy as np


class PoseSequence:
    # sequence是个npy文件
    def __init__(self, sequence):
        self.poses = []
        for parts in sequence:
            self.poses.append(Pose(parts))
        
        # normalize poses based on the average torso pixel length
        # 脖子和左臀距离
        torso_lengths = np.array([Part.dist(pose.neck, pose.lhip) for pose in self.poses if pose.neck.exists and pose.lhip.exists] +
                                 [Part.dist(pose.neck, pose.rhip) for pose in self.poses if pose.neck.exists and pose.rhip.exists])
        mean_torso = np.mean(torso_lengths)

        for pose in self.poses:
            for attr, part in pose:
                # 姿势，关节，值
                setattr(pose, attr, part / mean_torso)


class Pose:
    # PART_NAMES = ['nose', 'neck',  'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear']
    # ‘鼻子’、‘脖子’、‘肩’、‘肘’、‘腕’、‘肩’、‘肘’、‘腕’、‘髋’、‘膝’、‘踝’、、‘膝’、‘踝’、‘眼’、‘耳’]
    PART_NAMES=['nose', 'neck', 'rshoulder', 'relbow', 'rwrist',
                'lshoulder', 'lelbow', 'lwrist', 'midhip', 'rhip',
                'rknee', 'rankle', 'lhip', 'lknee', 'lankle',
                'reye', 'leye', 'rear', 'lear', 'lbigtoe',
                'lsmalltoe', 'lheel', 'rbigtoe', 'rsmalltoe', 'rheel']

    def __init__(self, parts):
        """Construct a pose for one frame, given an array of parts

        Arguments:
            parts - 25 * 3 ndarray of x, y, confidence values
        """
        # parts是一个25个关节点的骨架
        for name, vals in zip(self.PART_NAMES, parts):
            setattr(self, name, Part(vals))

    # 迭代器
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    # 字符串重写函数
    # getattr函数--返回对象属性
    def __str__(self):
        out = ""
        for name in self.PART_NAMES:
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out
    
    def print(self, parts):
        out = ""
        for name in parts:
            if not name in self.PART_NAMES:
                raise NameError(name)
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out


class Part:
    # x,y是坐标，c表示？
    def __init__(self, vals):
        self.x = vals[0]
        self.y = vals[1]
        self.c = vals[2]
        self.exists = self.c != 0.0

    def __truediv__(self, scalar):
        return Part([self.x / scalar, self.y / scalar, self.c])

    def __floordiv__(self, scalar):
        __truediv__(self, scalar)

    @staticmethod
    def dist(part1, part2):
        return np.sqrt(np.square(part1.x - part2.x) + np.square(part1.y - part2.y))

