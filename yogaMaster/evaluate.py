import os
import numpy as np


def evaluate_pose(pose_seq, exercise):
    """Evaluate a pose sequence for a particular exercise.

    Args:
        pose_seq: PoseSequence object.
        exercise: String name of the exercise to evaluate.

    Returns:
        correct: Bool whether exercise was performed correctly.
        feedback: Feedback string.

    """
    # 待改正，，将exercise改为标号，比如1，2，3.....后期排版完成后进行改正
    print('This is a '+exercise+' POSTURE.')
    if exercise == 'bicep_curl':
        return _bicep_curl(pose_seq)
    elif exercise == 'shoulder_press':
        return _shoulder_press(pose_seq)
    elif exercise == 'front_raise':
        return _front_raise(pose_seq)
    elif exercise == 'shoulder_shrug':
        return _shoulder_shrug(pose_seq)
    elif exercise == 'half_moon3':
        return _half_moon3(pose_seq)
    elif exercise == 'shushi':
        return _shushi(pose_seq)
    elif exercise == 'half_moon1':
        return _half_moon1(pose_seq)
    elif exercise == 'half_moon2':
        return _half_moon2(pose_seq)
    elif exercise == 'camle':
        return _camle(pose_seq)
    elif exercise == 'boat':
        return _boat(pose_seq)
    elif exercise == 'dance':
        return _dance(pose_seq)
    elif exercise == 'desk':
        return _desk(pose_seq)
    elif exercise == 'Vbalance':
        return _Vbalance(pose_seq)
    elif exercise == 'upside_angle':
        return _upside_angle(pose_seq)
    elif exercise == 'monkey':
        return _monkey(pose_seq)
    else:
        return (False, "Exercise string not recognized.")


def _monkey(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    right_present = [1 for pose in poses
                     if pose.rshoulder.exists and pose.relbow.exists and pose.rwrist]
    left_present = [1 for pose in poses
                    if pose.lshoulder.exists and pose.lelbow.exists and pose.lwrist]
    right_count = sum(right_present)
    left_count = sum(left_present)
    # ==为左边，左右算法有待改进
    side = 'right' if right_count > left_count else 'left'
    # print(right_count, left_count)
    # [print(pose.lshoulder.y, pose.rshoulder.y) for pose in poses]
    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rshoulder, pose.relbow, pose.midhip, pose.neck, pose.rhip, pose.rknee, pose.rankle) for pose in poses]
    else:
        joints = [(pose.lshoulder, pose.lelbow, pose.lwrist, pose.midhip, pose.neck, pose.rhip, pose.rknee, pose.rankle, pose.lhip, pose.lknee) for pose in poses]
    #                      0             1              2          3        4           5            6        7              8           9
    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]

    # <3 上手臂
    wrist_elbow_vecs = np.array([(joint[2].x - joint[1].x, joint[2].y - joint[1].y) for joint in joints])
    elbow_shoulder_vecs = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    # 主干--neck
    midhip_neck_vecs = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    # 腿弯<1--knee
    ankle_knee_vecs = np.array([(joint[6].x - joint[7].x, joint[6].y - joint[7].y) for joint in joints])
    hip_knee_vecs = np.array([(joint[6].x - joint[5].x, joint[6].y - joint[5].y) for joint in joints])
    # <2 midhip_neck_vecs+hip_knee_vecs
    # <4 midhip_neck_vecs+rehip_knee_vecs
    rehip_knee_vecs = np.array([(joint[9].x - joint[8].x, joint[9].y - joint[8].y) for joint in joints])

    # normalize vectors
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    midhip_neck_vecs = midhip_neck_vecs / np.expand_dims(np.linalg.norm(midhip_neck_vecs, axis=1), axis=1)
    elbow_shoulder_vecs = elbow_shoulder_vecs / np.expand_dims(np.linalg.norm(elbow_shoulder_vecs, axis=1), axis=1)
    ankle_knee_vecs = ankle_knee_vecs / np.expand_dims(np.linalg.norm(ankle_knee_vecs, axis=1), axis=1)
    wrist_elbow_vecs = wrist_elbow_vecs / np.expand_dims(np.linalg.norm(wrist_elbow_vecs, axis=1), axis=1)
    rehip_knee_vecs = rehip_knee_vecs / np.expand_dims(np.linalg.norm(rehip_knee_vecs, axis=1), axis=1)

    elbow_shoulder_wrist_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(wrist_elbow_vecs, elbow_shoulder_vecs), axis=1), -1.0, 1.0)))
    hip_knee_ankle_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, ankle_knee_vecs), axis=1), -1.0, 1.0)))
    midhip_neck_hip_knee_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(midhip_neck_vecs, hip_knee_vecs), axis=1), -1.0, 1.0)))
    midhip_neck_rehip_knee_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(midhip_neck_vecs, rehip_knee_vecs), axis=1), -1.0, 1.0)))

    correct = True
    feedback = ''
    # <3 上胳膊和上肘部太大
    if elbow_shoulder_wrist_angles > 40.0:
        correct = False
        feedback += 'Your arm is too far away from your body. Please keep your hands close to your body and spread your fingers. '+ \
                    'Keep your arm as close to your upper body as possible.'
    # <1右腿不够伸直
    if hip_knee_ankle_angles < 165.0:
        correct = False
        feedback += 'Your legs are not straight, please try to transfer the center to the abdomen, and do more leg exercises in the future.'
    # <2 主干和右肢体角度太小
    if midhip_neck_hip_knee_angles < 95.0:
        correct = False
        feedback +='Your upper body is not upright, please keep your body vertical to the ground as much as possible, '+ \
                   'and stretch the front leg as much as possible.'
    # <4主干和左肢体角度太小
    if midhip_neck_rehip_knee_angles < 100.0:
        correct = False
        feedback += 'Your upper body is not upright, please keep your body vertical to the ground as much as possible, '+ \
                    'and stretch your rear legs as much as possible. '
    if correct:
        return (
            correct, 'Exercise performed correctly! Weight was lifted fully.')
    else:
        return (correct, feedback)


def _upside_angle(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    right_present = [1 for pose in poses
                     if pose.lshoulder.x < pose.rshoulder.x]
    left_present = [1 for pose in poses
                    if pose.lshoulder.x > pose.rshoulder.x]
    right_count = sum(right_present)
    left_count = sum(left_present)
    # ==为左边，左右算法有待改进
    side = 'right' if right_count > left_count else 'left'
    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.lankle, pose.lknee, pose.lhip, pose.midhip, pose.neck, pose.rhip, pose.rknee, pose.rwrist,
                   pose.lshoulder, pose.lelbow)
                  for pose in poses]
    else:
        joints = [(pose.rankle, pose.rknee, pose.rhip, pose.midhip, pose.neck, pose.lhip, pose.lknee,pose.lwrist,pose.rshoulder,pose.relbow)
                  for pose in poses]
    #                  0             1              2          3           4           5          6            7               8
    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    # 四个角度，八个肢体向量
    # 下小腿
    ankle_knee_vecs_1 = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    # 上大腿
    hip_knee_vecs = np.array([(joint[2].x - joint[1].x, joint[2].y - joint[1].y) for joint in joints])
    # 脖子到中点,boat的角度+
    neck_midhip_vecs = np.array([(joint[3].x - joint[4].x, joint[3].y - joint[4].y) for joint in joints])
    # 左侧臀到左膝+
    knee_hip_vecs = np.array([(joint[5].x - joint[6].x, joint[5].y - joint[6].y) for joint in joints])
    # 左腕到左膝盖
    ankle_knee_vecs = np.array([(joint[7].x - joint[6].x, joint[7].y - joint[6].y) for joint in joints])
    # 肩部到肘部
    shoulder_elbow_vecs = np.array([(joint[9].x - joint[8].x, joint[9].y - joint[8].y) for joint in joints])
    # 脖子到肩部
    shoulder_neck_vecs = np.array([(joint[4].x - joint[8].x, joint[4].y - joint[8].y) for joint in joints])

    # 上臂和中支柱的角度，neck_midhip_vecs + shoulder_elbow_vecs

    # normalize vectors
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    ankle_knee_vecs_1 = ankle_knee_vecs_1 / np.expand_dims(np.linalg.norm(ankle_knee_vecs_1, axis=1), axis=1)
    neck_midhip_vecs = neck_midhip_vecs / np.expand_dims(np.linalg.norm(neck_midhip_vecs, axis=1), axis=1)
    knee_hip_vecs = knee_hip_vecs / np.expand_dims(np.linalg.norm(knee_hip_vecs, axis=1), axis=1)
    ankle_knee_vecs = ankle_knee_vecs / np.expand_dims(np.linalg.norm(ankle_knee_vecs, axis=1), axis=1)
    shoulder_elbow_vecs = shoulder_elbow_vecs / np.expand_dims(np.linalg.norm(shoulder_elbow_vecs, axis=1), axis=1)
    shoulder_neck_vecs = shoulder_neck_vecs / np.expand_dims(np.linalg.norm(shoulder_neck_vecs, axis=1), axis=1)

    hip_knee_angle_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, ankle_knee_vecs_1), axis=1), -1.0, 1.0)))
    neck_midhip_knee_hip_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(knee_hip_vecs, neck_midhip_vecs), axis=1), -1.0, 1.0)))
    ankle_knee_hip_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(knee_hip_vecs, ankle_knee_vecs), axis=1), -1.0, 1.0)))
    neck_shoulder_elbow_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(shoulder_elbow_vecs, shoulder_neck_vecs), axis=1), -1.0, 1.0)))
    print(hip_knee_angle_angles)
    print(neck_midhip_knee_hip_angles)
    print(ankle_knee_hip_angles)
    print(neck_shoulder_elbow_angles)
    correct = True
    feedback = ''
    # 腿没有伸展开来<1
    if hip_knee_angle_angles < 170.0:
        correct = False
        feedback += 'Your legs don''t seem to be straight up enough. Maybe you should try to take a bigger step and keep your legs straight.\n'
    # 上身和大腿角度太大<2
    if neck_midhip_knee_hip_angles > 40.0:
        correct = False
        feedback += 'your upper body doesn''t seem to bend down to the right position. You should make the hips '+ \
                    'and legs work, turn down as much as possible, and keep your balance.'
    # <3另一边的腿没有大于
    if ankle_knee_hip_angles > 115.0:
        correct = False
        feedback += 'your body doesn''t seem to squat down to the right position. You should make your hips and legs ' + \
                    'strong, bend your legs down as much as possible, and keep your body balanced.'
    # <4上臂小于
    if neck_shoulder_elbow_angles < 95.0:
        correct = False
        feedback += 'your arm extension needs to be improved. You should try to look up with your eyes inclined,' \
                    ' and extend your arm upward with the waist as the focus.'
    if correct:
        return (
            correct, 'Exercise performed correctly! Weight was lifted fully.')
    else:
        return (correct, feedback)


def _Vbalance(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    right_present = [1 for pose in poses
                     if pose.lshoulder.x < pose.lelbow.x]
    left_present = [1 for pose in poses
                    if pose.lshoulder.x > pose.lelbow.x]
    right_count = sum(right_present)
    left_count = sum(left_present)
    # ==为左边，左右算法有待改进
    side = 'right' if right_count > left_count else 'left'
    # print(right_count, left_count)
    # [print(pose.lshoulder.y, pose.rshoulder.y) for pose in poses]
    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rwrist, pose.relbow, pose.rshoulder, pose.rhip, pose.rknee, pose.rankle, pose.midhip, pose.neck)
                  for pose in poses]
    else:
        joints = [(pose.lwrist, pose.lelbow, pose.lshoulder, pose.lhip, pose.lknee, pose.lankle, pose.midhip, pose.neck)
                  for pose in poses]
    #                        0             1              2          3           4           5          6            7               8

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    # 四个角度，八个肢体向量
    # 上大腿
    hip_knee_vecs = np.array([(joint[3].x - joint[4].x, joint[3].y - joint[4].y) for joint in joints])
    # 下小腿
    ankle_knee_vecs = np.array([(joint[5].x - joint[4].x, joint[5].y - joint[4].y) for joint in joints])
    # 脖子到中点,boat的角度
    neck_midhip_vecs = np.array([(joint[7].x - joint[6].x, joint[7].y - joint[6].y) for joint in joints])
    knee_hip_vecs = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])# 另一侧中点到侧臀
    # 上直臂两向量
    wrist_elbow_vecs = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    shoulder_elbow_vecs = np.array([(joint[2].x - joint[1].x, joint[2].y - joint[1].y) for joint in joints])

    # 上臂和中支柱的角度，neck_midhip_vecs + shoulder_elbow_vecs

    # normalize vectors
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    ankle_knee_vecs = ankle_knee_vecs / np.expand_dims(np.linalg.norm(ankle_knee_vecs, axis=1), axis=1)
    neck_midhip_vecs = neck_midhip_vecs / np.expand_dims(np.linalg.norm(neck_midhip_vecs, axis=1), axis=1)
    knee_hip_vecs = knee_hip_vecs / np.expand_dims(np.linalg.norm(knee_hip_vecs, axis=1), axis=1)
    wrist_elbow_vecs = wrist_elbow_vecs / np.expand_dims(np.linalg.norm(wrist_elbow_vecs, axis=1), axis=1)
    shoulder_elbow_vecs = shoulder_elbow_vecs / np.expand_dims(np.linalg.norm(shoulder_elbow_vecs, axis=1), axis=1)

    hip_knee_angle_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, ankle_knee_vecs), axis=1), -1.0, 1.0)))
    neck_midhip_knee_hip_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_midhip_vecs, knee_hip_vecs), axis=1), -1.0, 1.0)))
    shoulder_elbow_wrist_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(wrist_elbow_vecs, shoulder_elbow_vecs), axis=1), -1.0, 1.0)))
    neck_midhip_shoulder_elbow_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_midhip_vecs, shoulder_elbow_vecs), axis=1), -1.0, 1.0)))

    correct = True
    feedback = ''
    # 腿没有伸展开来
    if hip_knee_angle_angles < 170.0:
        correct = False
        feedback += 'Your legs don''t seem to be straight up enough. Maybe you should try to take a bigger step and keep your legs straight.\n'
    # 上身和大腿角度太大
    if neck_midhip_knee_hip_angles > 100.0:
        correct = False
        feedback += 'Your legs don''t seem to be raised to the right position. ' + \
                    'You can try to lift your legs with the strength of your hips, which will be very useful for your leg muscles.'
    # 胳膊没有伸展开来
    if shoulder_elbow_wrist_angles < 120.0:
        correct = False
        feedback += 'Your arm may not be straight up. Please adjust the angle of arm extension to make ' + \
                    'it stretch up vertically as much as possible, which is conducive to the coordinated ' + \
                    'development of the body'
    # 上身和胳膊的角度太小
    if neck_midhip_shoulder_elbow_angles < 80.0:
        correct = False
        feedback += 'Your arms don''t seem to be raised to the right position. ' + \
                    'You can try to lift your arms with the strength of your shoulders, which will be very useful for your arms muscles.'
    if correct:
        return (
            correct, 'Exercise performed correctly! Weight was lifted fully.')
    else:
        return (correct, feedback)


def _desk(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    right_present = [1 for pose in poses
                     if pose.rshoulder.exists and pose.relbow.exists and pose.rhip.exists and pose.rknee.exists and pose.rankle.exists]
    left_present = [1 for pose in poses
                    if pose.lshoulder.exists and pose.lelbow.exists and pose.lhip.exists and pose.lknee.exists and pose.lankle.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    # ==为左边，左右算法有待改进
    side = 'right' if right_count > left_count else 'left'
    # print(right_count, left_count)
    # [print(pose.lshoulder.y, pose.rshoulder.y) for pose in poses]
    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rshoulder, pose.relbow, pose.midhip, pose.neck, pose.rhip, pose.rknee, pose.rankle) for pose in poses]
    else:
        joints = [(pose.lshoulder, pose.lelbow,pose.midhip, pose.neck, pose.lhip, pose.lknee, pose.lankle) for pose in poses]
    #                      0             1              2          3        4           5            6
    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]

    # <3
    elbow_shoulder_vecs = np.array([(joint[1].x - joint[0].x, joint[1].y - joint[0].y) for joint in joints])
    # 主干--neck
    midhip_neck_vecs = np.array([(joint[2].x - joint[3].x, joint[2].y - joint[3].y) for joint in joints])
    # 腿弯<1--knee
    hip_knee_vecs = np.array([(joint[4].x - joint[5].x, joint[4].y - joint[5].y) for joint in joints])
    ankle_knee_vecs = np.array([(joint[6].x - joint[5].x, joint[6].y - joint[5].y) for joint in joints])
    # <2 midhip_neck_vecs hip_knee_vecs

    # normalize vectors
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    midhip_neck_vecs = midhip_neck_vecs / np.expand_dims(np.linalg.norm(midhip_neck_vecs, axis=1), axis=1)
    elbow_shoulder_vecs = elbow_shoulder_vecs / np.expand_dims(np.linalg.norm(elbow_shoulder_vecs, axis=1), axis=1)
    ankle_knee_vecs = ankle_knee_vecs / np.expand_dims(np.linalg.norm(ankle_knee_vecs, axis=1), axis=1)

    elbow_shoulder_midhip_neck_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(elbow_shoulder_vecs, midhip_neck_vecs), axis=1), -1.0, 1.0)))
    hip_knee_ankle_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, ankle_knee_vecs), axis=1), -1.0, 1.0)))
    midhip_neck_hip_knee_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(midhip_neck_vecs, hip_knee_vecs), axis=1), -1.0, 1.0)))

    correct = True
    feedback = ''
    # <3 上胳膊和上肢体角度太小
    if elbow_shoulder_midhip_neck_angles < 80.0:
        correct = False
        feedback += 'Your upper body doesn''t seem to be supported high enough. ' + \
                    'Please try your arm. Push up and form a bow, keeping your body as parallel as possible to the ground.'
    # <1大小腿角度太小
    if hip_knee_ankle_angles < 93.0:
        correct = False
        feedback += 'Your body doesn''t seem to be supported high enough. Please try thigh force, body upward force, ' + \
                    'and keep your body parallel to the ground as much as possible.'
    # <2 大腿和主干角度不平
    if midhip_neck_hip_knee_angles < 160.0:
        correct = False
        feedback += 'Your body doesn''t seem to be supported high enough. Please try to use your belly force, ' + \
                    'push your body up and form a bow, and keep your body parallel to the ground as much as possible.            '
    if correct:
        return (
            correct, 'Exercise performed correctly! Weight was lifted fully.')
    else:
        return (correct, feedback)


def _dance(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    right_present = [1 for pose in poses
                     if pose.lshoulder.x < pose.lelbow.x]
    left_present = [1 for pose in poses
                    if pose.lshoulder.x > pose.lelbow.x]
    right_count = sum(right_present)
    left_count = sum(left_present)
    # ==为左边，左右算法有待改进
    side = 'right' if right_count > left_count else 'left'
    # print(right_count, left_count)
    # [print(pose.lshoulder.y, pose.rshoulder.y) for pose in poses]
    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.lhip, pose.lknee, pose.midhip, pose.neck, pose.rhip, pose.rknee, pose.rankle, pose.lshoulder,pose.lelbow)
                  for pose in poses]
    else:
        joints = [(pose.rhip, pose.rknee,pose.midhip, pose.neck, pose.lhip, pose.lknee, pose.lankle, pose.rshoulder, pose.relbow)
                  for pose in poses]
    #                0             1              2          3        4           5            6            7               8

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]

    # 右上大腿--knee <4
    hip_knee_vecs = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    # 主干--neck
    midhip_neck_vecs = np.array([(joint[2].x - joint[3].x, joint[2].y - joint[3].y) for joint in joints])
    # 腿弯<1--knee
    lhip_knee_vecs = np.array([(joint[4].x - joint[5].x, joint[4].y - joint[5].y) for joint in joints])
    wrist_knee_vecs = np.array([(joint[6].x - joint[5].x, joint[6].y - joint[5].y) for joint in joints])
    # <3
    shoulder_elbow_vecs = np.array([(joint[7].x - joint[8].x, joint[7].y - joint[8].y) for joint in joints])
    # hip_knee_vecs
    # normalize vectors
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    midhip_neck_vecs = midhip_neck_vecs / np.expand_dims(np.linalg.norm(midhip_neck_vecs, axis=1), axis=1)
    lhip_knee_vecs = lhip_knee_vecs / np.expand_dims(np.linalg.norm(lhip_knee_vecs, axis=1), axis=1)
    wrist_knee_vecs = wrist_knee_vecs / np.expand_dims(np.linalg.norm(wrist_knee_vecs, axis=1), axis=1)
    shoulder_elbow_vecs = shoulder_elbow_vecs / np.expand_dims(np.linalg.norm(shoulder_elbow_vecs, axis=1), axis=1)

    hip_knee_midhip_neck_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, midhip_neck_vecs), axis=1), -1.0, 1.0)))
    lhip_knee_wrist_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(lhip_knee_vecs, wrist_knee_vecs), axis=1), -1.0, 1.0)))
    elbow_shoulder_knee_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(shoulder_elbow_vecs, hip_knee_vecs), axis=1), -1.0, 1.0)))

    correct = True
    feedback = ''
    # <4 大腿和躯干太大
    if hip_knee_midhip_neck_angles > 145.0:
        correct = False
        feedback += 'Your body may not be leaning forward properly, or you should shift your body''s ' + \
                    'center of gravity to your hips, lean forward properly, and keep your balance.'
    # <1大小腿角度太大
    if lhip_knee_wrist_angles > 70.0:
        correct = False
        feedback += 'Your legs don''t seem to be raised to the right position, maybe you should ' +\
                    'lift them higher and bend them properly to keep your balance.'
    # <3胳膊和躯干大于垂直角度
    if elbow_shoulder_knee_angles > 107.0:
        correct = False
        feedback += 'Your arms are stretched too high. Maybe you should lower your arms properly, ' + \
                    'keep them parallel to the ground plane, and keep your body stable.'
    # <3胳膊和躯干小于垂直角度
    if elbow_shoulder_knee_angles < 73.0:
        correct = False
        feedback += 'Your arms are stretched too low, maybe you should lift them up properly, ' +\
                    'keep them parallel to the ground plane, and keep your body stable.'
    if correct:
        return (
            correct, 'Exercise performed correctly! Weight was lifted fully.')
    else:
        return (correct, feedback)


def _boat(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    right_present = [1 for pose in poses
                     if pose.lshoulder.x < pose.lelbow.x]
    left_present = [1 for pose in poses
                    if pose.lshoulder.x > pose.lelbow.x]
    right_count = sum(right_present)
    left_count = sum(left_present)
    # ==为左边，左右算法有待改进
    side = 'right' if right_count > left_count else 'left'
    # print(right_count, left_count)
    # [print(pose.lshoulder.y, pose.rshoulder.y) for pose in poses]
    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rwrist, pose.relbow, pose.rshoulder, pose.rhip, pose.rknee, pose.rankle, pose.midhip, pose.neck)
                  for pose in poses]
    else:
        joints = [(pose.lwrist, pose.lelbow, pose.lshoulder, pose.lhip, pose.lknee, pose.lankle, pose.midhip, pose.neck)
                  for pose in poses]
    #                        0             1              2          3           4           5          6            7               8

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    # 四个角度，八个肢体向量
    # 上大腿
    hip_knee_vecs = np.array([(joint[3].x - joint[4].x, joint[3].y - joint[4].y) for joint in joints])
    # 下小腿
    ankle_knee_vecs = np.array([(joint[5].x - joint[4].x, joint[5].y - joint[4].y) for joint in joints])
    # 脖子到中点,boat的角度
    neck_midhip_vecs = np.array([(joint[7].x - joint[6].x, joint[7].y - joint[6].y) for joint in joints])
    knee_hip_vecs = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])# 另一侧中点到侧臀
    # 上直臂两向量
    wrist_elbow_vecs = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    shoulder_elbow_vecs = np.array([(joint[2].x - joint[1].x, joint[2].y - joint[1].y) for joint in joints])

    # 上臂和中支柱的角度，neck_midhip_vecs + shoulder_elbow_vecs

    # normalize vectors
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    ankle_knee_vecs = ankle_knee_vecs / np.expand_dims(np.linalg.norm(ankle_knee_vecs, axis=1), axis=1)
    neck_midhip_vecs = neck_midhip_vecs / np.expand_dims(np.linalg.norm(neck_midhip_vecs, axis=1), axis=1)
    knee_hip_vecs = knee_hip_vecs / np.expand_dims(np.linalg.norm(knee_hip_vecs, axis=1), axis=1)
    wrist_elbow_vecs = wrist_elbow_vecs / np.expand_dims(np.linalg.norm(wrist_elbow_vecs, axis=1), axis=1)
    shoulder_elbow_vecs = shoulder_elbow_vecs / np.expand_dims(np.linalg.norm(shoulder_elbow_vecs, axis=1), axis=1)

    hip_knee_angle_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, ankle_knee_vecs), axis=1), -1.0, 1.0)))
    neck_midhip_knee_hip_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_midhip_vecs, knee_hip_vecs), axis=1), -1.0, 1.0)))
    shoulder_elbow_wrist_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(wrist_elbow_vecs, shoulder_elbow_vecs), axis=1), -1.0, 1.0)))
    neck_midhip_shoulder_elbow_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_midhip_vecs, shoulder_elbow_vecs), axis=1), -1.0, 1.0)))

    correct = True
    feedback = ''
    # 腿没有伸展开来
    if hip_knee_angle_angles < 170:
        correct = False
        feedback += 'Your legs don''t seem to be straight up enough. Maybe you should try to take a bigger step and keep your legs straight.\n'
    # 上身和大腿角度太大
    if neck_midhip_knee_hip_angles > 100:
        correct = False
        feedback += 'Your legs don''t seem to be raised to the right position. ' + \
                    'You can try to lift your legs with the strength of your hips, which will be very useful for your leg muscles.'
    # 胳膊没有伸展开来
    if shoulder_elbow_wrist_angles < 155.0:
        correct = False
        feedback += 'Your arm may not be straight up. Please adjust the angle of arm extension to make ' + \
                    'it stretch up vertically as much as possible, which is conducive to the coordinated ' + \
                    'development of the body'
    # 上身和胳膊的角度太小
    if neck_midhip_shoulder_elbow_angles < 50.0:
        correct = False
        feedback += 'Your arms don''t seem to be raised to the right position. ' + \
                    'You can try to lift your arms with the strength of your shoulders, which will be very useful for your arms muscles.'
    if correct:
        return (
            correct, 'Exercise performed correctly! Weight was lifted fully.')
    else:
        return (correct, feedback)


def _camle(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    # right_present = [1 for pose in poses
    #                  if pose.rknee.exists and pose.rhip.exists and pose.rankle.exists]
    # left_present = [1 for pose in poses
    #                 if pose.lknee.exists and pose.lhip.exists and pose.lankle.exists]
    right_present = [1 for pose in poses
                     if pose.rshoulder.y > pose.lshoulder.y]
    left_present = [1 for pose in poses
                    if pose.lshoulder.y > pose.rshoulder.y]
    right_count = sum(right_present)
    left_count = sum(left_present)
    # ==为左边，左右算法有待改进
    side = 'right' if right_count > left_count else 'left'
    # print(right_count, left_count)
    # [print(pose.lshoulder.y, pose.rshoulder.y) for pose in poses]
    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rknee, pose.rhip, pose.rankle, pose.midhip, pose.neck, pose.lknee, pose.lhip, pose.lshoulder, pose.lelbow) for pose in poses]
    else:
        joints = [(pose.lknee, pose.lhip, pose.lankle, pose.midhip, pose.neck, pose.rknee, pose.rhip, pose.rshoulder, pose.relbow) for pose in poses]
    #                        0         1            2            3          4           5          6            7               8

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    # 四个角度，八个肢体向量
    # 上大腿
    knee_rhip_vecs = np.array([(joint[5].x - joint[6].x, joint[5].y - joint[6].y) for joint in joints])
    # 中点到侧臀
    rhip_midhip_vecs = np.array([(joint[3].x - joint[6].x, joint[3].y - joint[6].y) for joint in joints])
    # 脖子到中点
    neck_midhip_vecs = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    lhip_midhip_vecs = np.array([(joint[1].x - joint[3].x, joint[1].y - joint[3].y) for joint in joints])
    # 另一侧中点到侧臀
    # 上直臂两向量
    neck_shoulder_vecs = np.array([(joint[4].x - joint[7].x, joint[4].y - joint[7].y) for joint in joints])
    shoulder_elbow_vecs = np.array([(joint[8].x - joint[7].x, joint[8].y - joint[7].y) for joint in joints])

    # 膝盖到臀部
    hip_knee_vecs = np.array([(joint[1].x - joint[0].x, joint[1].y - joint[0].y) for joint in joints])

    # 下肢
    ankle_knee_vecs = np.array([(joint[2].x - joint[0].x, joint[2].y - joint[0].y) for joint in joints])

    # normalize vectors
    knee_rhip_vecs = knee_rhip_vecs / np.expand_dims(np.linalg.norm(knee_rhip_vecs, axis=1), axis=1)
    rhip_midhip_vecs = rhip_midhip_vecs / np.expand_dims(np.linalg.norm(rhip_midhip_vecs, axis=1), axis=1)
    neck_midhip_vecs = neck_midhip_vecs / np.expand_dims(np.linalg.norm(neck_midhip_vecs, axis=1), axis=1)
    lhip_midhip_vecs = lhip_midhip_vecs / np.expand_dims(np.linalg.norm(lhip_midhip_vecs, axis=1), axis=1)
    neck_shoulder_vecs = neck_shoulder_vecs / np.expand_dims(np.linalg.norm(neck_shoulder_vecs, axis=1), axis=1)
    shoulder_elbow_vecs = shoulder_elbow_vecs / np.expand_dims(np.linalg.norm(shoulder_elbow_vecs, axis=1), axis=1)
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    ankle_knee_vecs = ankle_knee_vecs / np.expand_dims(np.linalg.norm(ankle_knee_vecs, axis=1), axis=1)

    midhip_hip_knee_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(knee_rhip_vecs, rhip_midhip_vecs), axis=1), -1.0, 1.0)))
    neck_midhip_hip_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_midhip_vecs, lhip_midhip_vecs), axis=1), -1.0, 1.0)))
    neck_shoulder_elbow_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_shoulder_vecs, shoulder_elbow_vecs), axis=1), -1.0, 1.0)))
    hip_knee_ankle_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, ankle_knee_vecs), axis=1), -1.0, 1.0)))

    # use thresholds learned from analysis
    # 视频所需
    # upper_arm_torso_range = np.max(upper_arm_torso_angles) - np.min(upper_arm_torso_angles)
    # upper_arm_forearm_min = np.min(upper_arm_forearm_angles)
    #
    # print('Upper arm and torso angle range: {}'.format(upper_arm_torso_range))
    # print('Upper arm and forearm minimum angle: {}'.format(upper_arm_forearm_min))

    # neck_midhip_hip_angles 和 hip_knee_ankle_angles 的问题
    # print(midhip_hip_knee_angles)
    # print(neck_midhip_hip_angles)
    # print(midhip_lhip_knee_angles)
    # print(hip_knee_ankle_angles)
    correct = True
    feedback = ''
    # 右腿没展开
    if midhip_hip_knee_angles < 83.0:
        correct = False
        feedback += 'Your legs don''t seem to be moving big enough. Maybe you should try to take a bigger step and keep your legs straight.\n'
    #  腰没下弯
    if neck_midhip_hip_angles > 52.0:
        correct = False
        feedback += 'Your waist may not bend down enough, you should bend down as much as possible while maintaining stability.\n'
    # 上直臂角度不对
    if neck_shoulder_elbow_angles < 155.0:
        correct = False
        feedback += 'Your arm may not be straight up. Please adjust the angle of arm extension to make ' + \
                    'it stretch up vertically as much as possible, which is conducive to the coordinated ' + \
                    'development of the body'
    # 左腿不直
    if hip_knee_ankle_angles < 169.0:
        correct = False
        feedback += 'Your legs may not be straight. Maybe you can do as many warm-up exercises or ' \
                    'leg exercises as you can, but be careful not to strain your legs.\n'
    if correct:
        return (
        correct, 'Exercise performed correctly! Weight was lifted fully.')
    else:
        return (correct, feedback)


def _half_moon1(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    # right_present = [1 for pose in poses
    #                  if pose.rknee.exists and pose.rhip.exists and pose.rankle.exists]
    # left_present = [1 for pose in poses
    #                 if pose.lknee.exists and pose.lhip.exists and pose.lankle.exists]
    right_present = [1 for pose in poses
                     if pose.rshoulder.y > pose.lshoulder.y]
    left_present = [1 for pose in poses
                    if pose.lshoulder.y > pose.rshoulder.y]
    right_count = sum(right_present)
    left_count = sum(left_present)
    # ==为左边，左右算法有待改进
    side = 'right' if right_count > left_count else 'left'
    # print(right_count, left_count)
    # [print(pose.lshoulder.y, pose.rshoulder.y) for pose in poses]
    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rknee, pose.rhip, pose.rankle, pose.midhip, pose.neck, pose.lknee, pose.lhip, pose.lshoulder, pose.lelbow) for pose in poses]
    else:
        joints = [(pose.lknee, pose.lhip, pose.lankle, pose.midhip, pose.neck, pose.rknee, pose.rhip, pose.rshoulder, pose.relbow) for pose in poses]
    #                        0         1            2            3          4           5          6            7               8

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    # 四个角度，八个肢体向量
    # 上大腿
    knee_rhip_vecs = np.array([(joint[5].x - joint[6].x, joint[5].y - joint[6].y) for joint in joints])
    # 中点到侧臀
    rhip_midhip_vecs = np.array([(joint[3].x - joint[6].x, joint[3].y - joint[6].y) for joint in joints])
    # 脖子到中点
    neck_midhip_vecs = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    lhip_midhip_vecs = np.array([(joint[1].x - joint[3].x, joint[1].y - joint[3].y) for joint in joints])
    # 另一侧中点到侧臀
    # 上直臂两向量
    neck_shoulder_vecs = np.array([(joint[4].x - joint[7].x, joint[4].y - joint[7].y) for joint in joints])
    shoulder_elbow_vecs = np.array([(joint[8].x - joint[7].x, joint[8].y - joint[7].y) for joint in joints])

    # 膝盖到臀部
    hip_knee_vecs = np.array([(joint[1].x - joint[0].x, joint[1].y - joint[0].y) for joint in joints])

    # 下肢
    ankle_knee_vecs = np.array([(joint[2].x - joint[0].x, joint[2].y - joint[0].y) for joint in joints])

    # normalize vectors
    knee_rhip_vecs = knee_rhip_vecs / np.expand_dims(np.linalg.norm(knee_rhip_vecs, axis=1), axis=1)
    rhip_midhip_vecs = rhip_midhip_vecs / np.expand_dims(np.linalg.norm(rhip_midhip_vecs, axis=1), axis=1)
    neck_midhip_vecs = neck_midhip_vecs / np.expand_dims(np.linalg.norm(neck_midhip_vecs, axis=1), axis=1)
    lhip_midhip_vecs = lhip_midhip_vecs / np.expand_dims(np.linalg.norm(lhip_midhip_vecs, axis=1), axis=1)
    neck_shoulder_vecs = neck_shoulder_vecs / np.expand_dims(np.linalg.norm(neck_shoulder_vecs, axis=1), axis=1)
    shoulder_elbow_vecs = shoulder_elbow_vecs / np.expand_dims(np.linalg.norm(shoulder_elbow_vecs, axis=1), axis=1)
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    ankle_knee_vecs = ankle_knee_vecs / np.expand_dims(np.linalg.norm(ankle_knee_vecs, axis=1), axis=1)

    midhip_hip_knee_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(knee_rhip_vecs, rhip_midhip_vecs), axis=1), -1.0, 1.0)))
    neck_midhip_hip_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_midhip_vecs, lhip_midhip_vecs), axis=1), -1.0, 1.0)))
    neck_shoulder_elbow_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_shoulder_vecs, shoulder_elbow_vecs), axis=1), -1.0, 1.0)))
    hip_knee_ankle_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, ankle_knee_vecs), axis=1), -1.0, 1.0)))

    # use thresholds learned from analysis
    # 视频所需
    # upper_arm_torso_range = np.max(upper_arm_torso_angles) - np.min(upper_arm_torso_angles)
    # upper_arm_forearm_min = np.min(upper_arm_forearm_angles)
    #
    # print('Upper arm and torso angle range: {}'.format(upper_arm_torso_range))
    # print('Upper arm and forearm minimum angle: {}'.format(upper_arm_forearm_min))

    # neck_midhip_hip_angles 和 hip_knee_ankle_angles 的问题
    # print(midhip_hip_knee_angles)
    # print(neck_midhip_hip_angles)
    # print(midhip_lhip_knee_angles)
    # print(hip_knee_ankle_angles)
    correct = True
    feedback = ''
    # 右腿没展开
    if midhip_hip_knee_angles < 83.0:
        correct = False
        feedback += 'Your legs don''t seem to be moving big enough. Maybe you should try to take a bigger step and keep your legs straight.\n'
    #  腰没下弯
    if neck_midhip_hip_angles > 52.0:
        correct = False
        feedback += 'Your waist may not bend down enough, you should bend down as much as possible while maintaining stability.\n'
    # 上直臂角度不对
    if neck_shoulder_elbow_angles < 155.0:
        correct = False
        feedback += 'Your arm may not be straight up. Please adjust the angle of arm extension to make ' + \
                    'it stretch up vertically as much as possible, which is conducive to the coordinated ' + \
                    'development of the body'
    # 左腿不直
    if hip_knee_ankle_angles < 169.0:
        correct = False
        feedback += 'Your legs may not be straight. Maybe you can do as many warm-up exercises or ' \
                    'leg exercises as you can, but be careful not to strain your legs.\n'
    if correct:
        return (
        correct, 'Exercise performed correctly! Weight was lifted fully.')
    else:
        return (correct, feedback)


def _half_moon2(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    # right_present = [1 for pose in poses
    #                  if pose.rknee.exists and pose.rhip.exists and pose.rankle.exists]
    # left_present = [1 for pose in poses
    #                 if pose.lknee.exists and pose.lhip.exists and pose.lankle.exists]
    right_present = [1 for pose in poses
                     if pose.rshoulder.y > pose.lshoulder.y]
    left_present = [1 for pose in poses
                    if pose.lshoulder.y > pose.rshoulder.y]
    right_count = sum(right_present)
    left_count = sum(left_present)
    # ==为左边，左右算法有待改进
    side = 'right' if right_count > left_count else 'left'
    # print(right_count, left_count)
    # [print(pose.lshoulder.y, pose.rshoulder.y) for pose in poses]
    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rknee, pose.rhip, pose.rankle, pose.midhip, pose.neck, pose.lknee, pose.lhip, pose.lshoulder, pose.lelbow) for pose in poses]
    else:
        joints = [(pose.lknee, pose.lhip, pose.lankle, pose.midhip, pose.neck, pose.rknee, pose.rhip, pose.rshoulder, pose.relbow) for pose in poses]
    #                        0         1            2            3          4           5          6            7               8

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    # 四个角度，八个肢体向量
    # 上大腿
    knee_rhip_vecs = np.array([(joint[5].x - joint[6].x, joint[5].y - joint[6].y) for joint in joints])
    # 中点到侧臀
    rhip_midhip_vecs = np.array([(joint[3].x - joint[6].x, joint[3].y - joint[6].y) for joint in joints])
    # 脖子到中点
    neck_midhip_vecs = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    lhip_midhip_vecs = np.array([(joint[1].x - joint[3].x, joint[1].y - joint[3].y) for joint in joints])
    # 另一侧中点到侧臀
    # 上直臂两向量
    neck_shoulder_vecs = np.array([(joint[4].x - joint[7].x, joint[4].y - joint[7].y) for joint in joints])
    shoulder_elbow_vecs = np.array([(joint[8].x - joint[7].x, joint[8].y - joint[7].y) for joint in joints])

    # 膝盖到臀部
    hip_knee_vecs = np.array([(joint[1].x - joint[0].x, joint[1].y - joint[0].y) for joint in joints])

    # 下肢
    ankle_knee_vecs = np.array([(joint[2].x - joint[0].x, joint[2].y - joint[0].y) for joint in joints])

    # normalize vectors
    knee_rhip_vecs = knee_rhip_vecs / np.expand_dims(np.linalg.norm(knee_rhip_vecs, axis=1), axis=1)
    rhip_midhip_vecs = rhip_midhip_vecs / np.expand_dims(np.linalg.norm(rhip_midhip_vecs, axis=1), axis=1)
    neck_midhip_vecs = neck_midhip_vecs / np.expand_dims(np.linalg.norm(neck_midhip_vecs, axis=1), axis=1)
    lhip_midhip_vecs = lhip_midhip_vecs / np.expand_dims(np.linalg.norm(lhip_midhip_vecs, axis=1), axis=1)
    neck_shoulder_vecs = neck_shoulder_vecs / np.expand_dims(np.linalg.norm(neck_shoulder_vecs, axis=1), axis=1)
    shoulder_elbow_vecs = shoulder_elbow_vecs / np.expand_dims(np.linalg.norm(shoulder_elbow_vecs, axis=1), axis=1)
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    ankle_knee_vecs = ankle_knee_vecs / np.expand_dims(np.linalg.norm(ankle_knee_vecs, axis=1), axis=1)

    midhip_hip_knee_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(knee_rhip_vecs, rhip_midhip_vecs), axis=1), -1.0, 1.0)))
    neck_midhip_hip_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_midhip_vecs, lhip_midhip_vecs), axis=1), -1.0, 1.0)))
    neck_shoulder_elbow_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_shoulder_vecs, shoulder_elbow_vecs), axis=1), -1.0, 1.0)))
    hip_knee_ankle_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, ankle_knee_vecs), axis=1), -1.0, 1.0)))

    # use thresholds learned from analysis
    # 视频所需
    # upper_arm_torso_range = np.max(upper_arm_torso_angles) - np.min(upper_arm_torso_angles)
    # upper_arm_forearm_min = np.min(upper_arm_forearm_angles)
    #
    # print('Upper arm and torso angle range: {}'.format(upper_arm_torso_range))
    # print('Upper arm and forearm minimum angle: {}'.format(upper_arm_forearm_min))

    # neck_midhip_hip_angles 和 hip_knee_ankle_angles 的问题
    # print(midhip_hip_knee_angles)
    # print(neck_midhip_hip_angles)
    # print(midhip_lhip_knee_angles)
    # print(hip_knee_ankle_angles)
    correct = True
    feedback = ''
    # 右腿没展开
    if midhip_hip_knee_angles < 105.0:
        correct = False
        feedback += 'Your legs don''t seem to be moving big enough. Maybe you should try to take a bigger step and keep your legs straight up.\n'
    #  腰没下弯
    if neck_midhip_hip_angles > 89.0:
        correct = False
        feedback += 'Your waist may not bend down enough, you should bend down as much as possible while maintaining stability.\n'
    # 上直臂角度不对
    if neck_shoulder_elbow_angles < 150.0:
        correct = False
        feedback += 'Your arm may not be straight up. Please adjust the angle of arm extension to make ' + \
                    'it stretch up vertically as much as possible, which is conducive to the coordinated ' + \
                    'development of the body'
    # 左腿不直
    if hip_knee_ankle_angles < 169.0:
        correct = False
        feedback += 'Your legs may not be straight. Maybe you can do as many warm-up exercises or ' \
                    'leg exercises as you can, but be careful not to strain your legs.\n'
    if correct:
        return (
        correct, 'Exercise performed correctly! Weight was lifted fully.')
    else:
        return (correct, feedback)


def _half_moon3(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    # right_present = [1 for pose in poses
    #                  if pose.rknee.exists and pose.rhip.exists and pose.rankle.exists]
    # left_present = [1 for pose in poses
    #                 if pose.lknee.exists and pose.lhip.exists and pose.lankle.exists]
    right_present = [1 for pose in poses
                     if pose.rshoulder.y > pose.lshoulder.y]
    left_present = [1 for pose in poses
                    if pose.lshoulder.y > pose.rshoulder.y]
    right_count = sum(right_present)
    left_count = sum(left_present)
    # ==为左边，左右算法有待改进
    side = 'right' if right_count > left_count else 'left'
    # print(right_count, left_count)
    # [print(pose.lshoulder.y, pose.rshoulder.y) for pose in poses]
    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rknee, pose.rhip, pose.rankle, pose.midhip, pose.neck, pose.lknee, pose.lhip, pose.lankle) for pose in poses]
    else:
        joints = [(pose.lknee, pose.lhip, pose.lankle, pose.midhip, pose.neck, pose.rknee, pose.rhip, pose.rankle,) for pose in poses]
    #                        0         1            2            3          4           5          6            7
    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    # 四个角度，八个肢体向量
    # 上大腿
    knee_rhip_vecs = np.array([(joint[5].x - joint[6].x, joint[5].y - joint[6].y) for joint in joints])
    # 中点到侧臀
    rhip_midhip_vecs = np.array([(joint[3].x - joint[6].x, joint[3].y - joint[6].y) for joint in joints])
    # 脖子到中点
    neck_midhip_vecs = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    # 另一侧中点到侧臀
    lhip_midhip_vecs = np.array([(joint[1].x - joint[3].x, joint[1].y - joint[3].y) for joint in joints])

    # 膝盖到臀部
    hip_knee_vecs = np.array([(joint[1].x - joint[0].x, joint[1].y - joint[0].y) for joint in joints])

    # 下肢
    ankle_knee_vecs = np.array([(joint[2].x - joint[0].x, joint[2].y - joint[0].y) for joint in joints])

    # normalize vectors
    knee_rhip_vecs = knee_rhip_vecs / np.expand_dims(np.linalg.norm(knee_rhip_vecs, axis=1), axis=1)
    rhip_midhip_vecs = rhip_midhip_vecs / np.expand_dims(np.linalg.norm(rhip_midhip_vecs, axis=1), axis=1)
    neck_midhip_vecs = neck_midhip_vecs / np.expand_dims(np.linalg.norm(neck_midhip_vecs, axis=1), axis=1)
    lhip_midhip_vecs = lhip_midhip_vecs / np.expand_dims(np.linalg.norm(lhip_midhip_vecs, axis=1), axis=1)
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    ankle_knee_vecs = ankle_knee_vecs / np.expand_dims(np.linalg.norm(ankle_knee_vecs, axis=1), axis=1)

    midhip_hip_knee_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(knee_rhip_vecs, rhip_midhip_vecs), axis=1), -1.0, 1.0)))
    neck_midhip_hip_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(neck_midhip_vecs, lhip_midhip_vecs), axis=1), -1.0, 1.0)))
    midhip_lhip_knee_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(lhip_midhip_vecs, hip_knee_vecs), axis=1), -1.0, 1.0)))
    hip_knee_ankle_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, ankle_knee_vecs), axis=1), -1.0, 1.0)))

    # use thresholds learned from analysis
    # 视频所需
    # upper_arm_torso_range = np.max(upper_arm_torso_angles) - np.min(upper_arm_torso_angles)
    # upper_arm_forearm_min = np.min(upper_arm_forearm_angles)
    #
    # print('Upper arm and torso angle range: {}'.format(upper_arm_torso_range))
    # print('Upper arm and forearm minimum angle: {}'.format(upper_arm_forearm_min))

    # neck_midhip_hip_angles 和 hip_knee_ankle_angles 的问题
    # print(midhip_hip_knee_angles)
    # print(neck_midhip_hip_angles)
    # print(midhip_lhip_knee_angles)
    # print(hip_knee_ankle_angles)
    correct = True
    feedback = ''
    # 右腿没展开
    if midhip_hip_knee_angles < 83.0:
        correct = False
        feedback += 'Your legs don''t seem to be moving big enough. Maybe you should try to take a bigger step and keep your legs straight.\n'
    #  腰没下弯
    if neck_midhip_hip_angles > 52.0:
        correct = False
        feedback += 'Your waist may not bend down enough, you should bend down as much as possible while maintaining stability.\n'
    # 左腿没迈出去
    if midhip_lhip_knee_angles < 155.0:
        correct = False
        feedback += 'Your legs don''t seem to be moving big enough. Maybe you should try to take a bigger step and keep your legs straight..\n'
    # 左腿不直
    if hip_knee_ankle_angles < 172.0:
        correct = False
        feedback += 'Your legs may not be straight. Maybe you can do as many warm-up exercises or ' \
                    'leg exercises as you can, but be careful not to strain your legs.\n'
    if correct:
        return (
        correct, 'Exercise performed correctly! Weight was lifted fully.')
    else:
        return (correct, feedback)


def _shushi(pose_seq):
    # find the arm that is seen most consistently
    # poses为很多个姿势列表，根据json文件而定
    poses = pose_seq.poses
    right_present = [1 for pose in poses
                     if pose.rshoulder.exists and pose.relbow.exists and pose.rhip.exists and pose.rknee.exists and pose.rankle.exists]
    left_present = [1 for pose in poses
                    if pose.lshoulder.exists and pose.lelbow.exists and pose.lhip.exists and pose.lknee.exists and pose.lankle.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    side = 'left' if right_count < left_count else 'right'

    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rshoulder, pose.relbow, pose.rknee, pose.rhip, pose.neck, pose.rankle, pose.midhip) for pose in poses]
    else:
        joints = [(pose.lshoulder, pose.lelbow, pose.lknee, pose.lhip, pose.neck, pose.lankle, pose.midhip) for pose in poses]
    #              0               1            2           3          4          5            6
    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    # 上胳膊向量
    upper_arm_vecs = np.array([(joint[1].x - joint[0].x, joint[1].y - joint[0].y) for joint in joints])
    # 肩部到脖子的向量
    shoulder_neck_vecs = np.array([(joint[4].x - joint[0].x, joint[4].y - joint[0].y) for joint in joints])
    # 身体中心躯干向量
    torso_vecs = np.array([(joint[4].x - joint[6].x, joint[4].y - joint[6].y) for joint in joints])
    # 中心点到臀的横向量
    mid_hip_vecs = np.array([(joint[3].x - joint[6].x, joint[3].y - joint[6].y) for joint in joints])
    # 大腿肢体
    hip_knee_vecs = np.array([(joint[3].x - joint[2].x, joint[3].y - joint[2].y) for joint in joints])
    # 小腿肢体
    ankle_knee_vecs = np.array([(joint[5].x - joint[2].x, joint[5].y - joint[2].y) for joint in joints])

    # normalize vectors
    upper_arm_vecs = upper_arm_vecs / np.expand_dims(np.linalg.norm(upper_arm_vecs, axis=1), axis=1)
    torso_vecs = torso_vecs / np.expand_dims(np.linalg.norm(torso_vecs, axis=1), axis=1)
    mid_hip_vecs = mid_hip_vecs / np.expand_dims(np.linalg.norm(mid_hip_vecs, axis=1), axis=1)
    hip_knee_vecs = hip_knee_vecs / np.expand_dims(np.linalg.norm(hip_knee_vecs, axis=1), axis=1)
    ankle_knee_vecs = ankle_knee_vecs / np.expand_dims(np.linalg.norm(ankle_knee_vecs, axis=1), axis=1)
    shoulder_neck_vecs = shoulder_neck_vecs / np.expand_dims(np.linalg.norm(shoulder_neck_vecs, axis=1), axis=1)

    elbow_shoulder_neck_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, shoulder_neck_vecs), axis=1), -1.0, 1.0)))
    neck_midhip_hip_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(torso_vecs, mid_hip_vecs), axis=1), -1.0, 1.0)))
    midhip_hip_ankle_angles = np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(hip_knee_vecs, ankle_knee_vecs), axis=1), -1.0, 1.0)))

    # use thresholds learned from analysis
    # 视频中存在范围
    # upper_arm_torso_range = np.max(upper_arm_torso_angles) - np.min(upper_arm_torso_angles)
    # upper_arm_forearm_min = np.min(upper_arm_forearm_angles)
    #
    # print('Upper arm and torso angle range: {}'.format(upper_arm_torso_range))
    # print('Upper arm and forearm minimum angle: {}'.format(upper_arm_forearm_min))

    correct = True
    feedback = ''

    if elbow_shoulder_neck_angles > 99.0:
        correct = False
        feedback += 'Your upper arm is up, but it''s too far from your head.' + \
                    'Try to make your upper arm as perpendicular to your shoulder as possible.\n'

    if neck_midhip_hip_angles < 85.0:
        correct = False
        feedback += 'Your upper torso doesn''t seem to stand straight. Maybe you need more practice to improve your stability until you stand straight.\n'

    if midhip_hip_ankle_angles > 35.0:
        correct = False
        feedback += 'Your lower leg doesn''t seem to be bent enough. It''s a bit difficult for beginners to reach the standard. ' + \
                                                                      'Try to practice to increase the flexibility of your lower leg.\n'
    if correct:
        return (
        correct, 'Exercise performed correctly! Weight was lifted fully up, and lower leg shows perfectly!.')
    else:
        return (correct, feedback)


def _bicep_curl(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    right_present = [1 for pose in poses 
            if pose.rshoulder.exists and pose.relbow.exists and pose.rwrist.exists]
    left_present = [1 for pose in poses
            if pose.lshoulder.exists and pose.lelbow.exists and pose.lwrist.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    side = 'right' if right_count > left_count else 'left'

    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.neck) for pose in poses]
    else:
        joints = [(pose.lshoulder, pose.lelbow, pose.lwrist, pose.lhip, pose.neck) for pose in poses]

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    # 上肢
    upper_arm_vecs = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    # 身体躯干
    torso_vecs = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    # 下肢
    forearm_vecs = np.array([(joint[2].x - joint[1].x, joint[2].y - joint[1].y) for joint in joints])

    # normalize vectors
    upper_arm_vecs = upper_arm_vecs / np.expand_dims(np.linalg.norm(upper_arm_vecs, axis=1), axis=1)
    torso_vecs = torso_vecs / np.expand_dims(np.linalg.norm(torso_vecs, axis=1), axis=1)
    forearm_vecs = forearm_vecs / np.expand_dims(np.linalg.norm(forearm_vecs, axis=1), axis=1)

    upper_arm_torso_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, torso_vecs), axis=1), -1.0, 1.0)))
    upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, forearm_vecs), axis=1), -1.0, 1.0)))

    # use thresholds learned from analysis
    upper_arm_torso_range = np.max(upper_arm_torso_angles) - np.min(upper_arm_torso_angles)
    upper_arm_forearm_min = np.min(upper_arm_forearm_angles)

    print('Upper arm and torso angle range: {}'.format(upper_arm_torso_range))
    print('Upper arm and forearm minimum angle: {}'.format(upper_arm_forearm_min))

    correct = True
    feedback = ''

    if upper_arm_torso_range > 35.0:
        correct = False
        feedback += 'Your upper arm shows significant rotation around the shoulder when curling. Try holding your upper arm still, parallel to your chest, ' + \
                    'and concentrate on rotating around your elbow only.\n'
    
    if upper_arm_forearm_min > 70.0:
        correct = False
        feedback += 'You are not curling the weight all the way to the top, up to your shoulders. Try to curl your arm completely so that your forearm is parallel with your torso. It may help to use lighter weight.\n'

    if correct:
        return (correct, 'Exercise performed correctly! Weight was lifted fully up, and upper arm did not move significantly.')
    else:
        return (correct, feedback)


def _front_raise(pose_seq):
    poses = pose_seq.poses

    right_present = [1 for pose in poses 
            if pose.rshoulder.exists and pose.relbow.exists and pose.rwrist.exists]
    left_present = [1 for pose in poses
            if pose.lshoulder.exists and pose.lelbow.exists and pose.lwrist.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    side = 'right' if right_count > left_count else 'left'

    print('Exercise arm detected as: {}.'.format(side))
    
    if side == 'right':
        joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.neck) for pose in poses]
    else:
        joints = [(pose.lshoulder, pose.lelbow, pose.lwrist, pose.lhip, pose.neck) for pose in poses]

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    joints = np.array(joints)
    
    # Neck to hip
    back_vec = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    # Check range of motion of the back
    back_vec_range = np.max(back_vec, axis=0) - np.min(back_vec, axis=0)
    print("Horizontal range of motion for back: %s" % back_vec_range[0])
    
    # Shoulder to hip    
    torso_vecs = np.array([(joint[0].x - joint[3].x, joint[0].y - joint[3].y) for joint in joints])
    # Arm
    arm_vecs = np.array([(joint[0].x - joint[2].x, joint[0].y - joint[2].y) for joint in joints])
    
    # normalize vectors
    torso_vecs = torso_vecs / np.expand_dims(np.linalg.norm(torso_vecs, axis=1), axis=1)
    arm_vecs = arm_vecs / np.expand_dims(np.linalg.norm(arm_vecs, axis=1), axis=1)
    
    # Check if raised all the way up
    angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(torso_vecs, arm_vecs), axis=1), -1.0, 1.0)))
    print("Max angle between torso and arm when lifting: ", np.max(angles))

    correct = True
    feedback = ''

    if back_vec_range[0] > 0.3:
        correct = False
        feedback += 'Your back shows significant movement. Try keeping your back straight and still when you lift the weight. Consider using lighter weight.\n'

    if np.max(angles) < 90.0:
        correct = False
        feedback += 'You are not lifting the weight all the way up. Finish with wrists at or slightly above shoulder level.\n'

    if correct:
        return (correct, 'Exercise performed correctly! Weight was lifted fully up, and no significant back movement was detected.')
    else:
        return (correct, feedback)


def _shoulder_shrug(pose_seq):
    poses = pose_seq.poses

    joints = [(pose.lshoulder, pose.rshoulder, pose.lelbow, pose.relbow, pose.lwrist, pose.rwrist) for pose in poses]

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    joints = np.array(joints)
    
    # Shoulder position
    shoulders = np.array([(joint[0].y, joint[1].y) for joint in joints])

    # Straining back
    shoulder_range = np.max(shoulders, axis=0) - np.min(shoulders, axis=0)
    print("Range of motion for shoulders: %s" % np.average(shoulder_range))
    
    # Shoulder to elbow    
    upper_arm_vecs = np.array([(joint[0].x - joint[2].x, joint[0].y - joint[2].y) for joint in joints])
    # Elbow to wrist
    forearm_vecs = np.array([(joint[2].x - joint[4].x, joint[2].y - joint[4].y) for joint in joints])
    
    # normalize vectors
    # np.linalg.norm(upper_arm_vecs, axis=1):求多个行的平方和开根号
    upper_arm_vecs = upper_arm_vecs / np.expand_dims(np.linalg.norm(upper_arm_vecs, axis=1), axis=1)
    forearm_vecs = forearm_vecs / np.expand_dims(np.linalg.norm(forearm_vecs, axis=1), axis=1)
    
    # Check if raised all the way up
    # np.sum(np.multiply(upper_arm_vecs, forearm_vecs), axis=1按行求和。
    # np.clip(,-1,1)最小值为-1，最大值为1，进行裁剪。
    upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, forearm_vecs), axis=1), -1.0, 1.0)))
    upper_forearm_angle = np.max(upper_arm_forearm_angles)
    print("Max upper arm and forearm angle: ", upper_forearm_angle)

    correct = True
    feedback = ''

    if np.average(shoulder_range) < 0.1:
        correct = False
        feedback += 'Your shoulders do not go through enough motion. Squeeze and raise your shoulders more through the exercise.\n'

    if upper_forearm_angle > 30.0:
        correct = False
        feedback += 'Your arms are bending when lifting. Keep your arms straight and still, and focus on moving only the shoulders.\n'

    if correct:
        return (correct, 'Exercise performed correctly! Shoulders went through full range of motion, and arms remained straight.')
    else:
        return (correct, feedback) 


def _shoulder_press(pose_seq):
    poses = pose_seq.poses
    
    right_present = [1 for pose in poses 
            if pose.rshoulder.exists and pose.relbow.exists and pose.rwrist.exists]
    left_present = [1 for pose in poses
            if pose.lshoulder.exists and pose.lelbow.exists and pose.lwrist.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    side = 'right' if right_count > left_count else 'left'

    print('Exercise arm detected as: {}.'.format(side))
    
    if side == 'right':
        joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.neck) for pose in poses]
    else:
        joints = [(pose.lshoulder, pose.lelbow, pose.lwrist, pose.lhip, pose.neck) for pose in poses]

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    joints_ = np.array(joints)
    
    # Neck to hip
    back_vec = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    # Check range of motion of the back
    # Straining back
    back_vec_range = np.max(back_vec, axis=0) - np.min(back_vec, axis=0)
    print("Range of motion for back: %s" % back_vec_range[0])
    
    # Rolling shoulder too much
    elbow = joints_[:, 1]
    elbow_x = np.array([joint.x for joint in elbow])

    neck = joints_[:, 4]
    neck_x = np.array([joint.x for joint in neck])
    elbow_neck_dist = 0 
    if side =='right':
        elbow_neck_dist = np.min(elbow_x - neck_x)
        print("Minimum distance between elbow and neck: ", np.min(elbow_x - neck_x))
    else:
        elbow_neck_dist = np.min(neck_x - elbow_x)
        print("Minimum distance between elbow and neck: ", np.min(neck_x - elbow_x))
    
    # Shoulder to elbow    
    upper_arm_vecs = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    # Elbow to wrist
    forearm_vecs = np.array([(joint[2].x - joint[1].x, joint[2].y - joint[1].y) for joint in joints])
    
    # normalize vectors
    upper_arm_vecs = upper_arm_vecs / np.expand_dims(np.linalg.norm(upper_arm_vecs, axis=1), axis=1)
    forearm_vecs = forearm_vecs / np.expand_dims(np.linalg.norm(forearm_vecs, axis=1), axis=1)
    
    # Check if raised all the way up
    upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, forearm_vecs), axis=1), -1.0, 1.0)))
    upper_forearm_angle = np.max(upper_arm_forearm_angles)
    print("Max upper arm and forearm angle: ", np.max(upper_arm_forearm_angles))

    correct = True
    feedback = ''

    if back_vec_range[0] > 0.16:
        correct = False
        feedback += 'Your back shows significant movement while pressing. Try keeping your back straight and still when you lift the weight.\n'
    
    if elbow_neck_dist < -0.12:
        correct = False
        feedback += 'You are rolling your shoulders when you lift the weights. Try to steady your shoulders and keep them parallel.\n'
    
    if upper_forearm_angle < 178:
        correct = False
        feedback += 'You are not lifting the weight all the way up. Extend your arms through the full range of motion. Lower the weight if necessary.\n'

    if correct:
        return (correct, 'Exercise performed correctly! Weight was lifted fully up, shoulders remained parallel, and no significant back movement was detected.')
    else:
        return (correct, feedback)