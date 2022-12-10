import babyai
from gym_minigrid.wrappers import *
env = gym.make('BabyAI-UnblockPickup-v0')
print(env.reset()['image'].shape)


from babyai.levels import level_dict
level_list = [name for name, level in level_dict.items()
              if (not getattr(level, 'is_bonus', False) and not name == 'MiniBossLevel')]
# print(level_list)
# ['BossLevel', 'BossLevelNoUnlock', 'GoTo', 'GoToImpUnlock', 
# 'GoToLocal', 'GoToLocalS5N2', 'GoToLocalS6N2', 'GoToLocalS6N3',
#  'GoToLocalS6N4', 'GoToLocalS7N4', 'GoToLocalS7N5', 'GoToLocalS8N2', 
# 'GoToLocalS8N3', 'GoToLocalS8N4', 'GoToLocalS8N5', 'GoToLocalS8N6', 
# 'GoToLocalS8N7', 'GoToObj', 'GoToObjMaze', 'GoToObjMazeOpen', 
# 'GoToObjMazeS4', 'GoToObjMazeS4R2', 'GoToObjMazeS5', 'GoToObjMazeS6',
#  'GoToObjMazeS7', 'GoToObjS4', 'GoToObjS6', 'GoToOpen', 'GoToRedBall', 
# 'GoToRedBallGrey', 'GoToRedBallNoDists', 'GoToSeq', 'GoToSeqS5R2', 
# 
# 'Open', 'Pickup', 'PickupLoc', 'PutNext', 
# 'PutNextLocal', 'PutNextLocalS5N3', 'PutNextLocalS6N4', 
# 'Synth', 'SynthLoc', 'SynthS5R2', 'SynthSeq', 'UnblockPickup', 'Unlock', 
# 'TestGoToBlocked', 'TestLotsOfBlockers', 'TestPutNextCloseToDoor',
#  'TestPutNextToBlocked', 'TestPutNextToCloseToDoor1', 'TestPutNextToCloseToDoor2', 
# 'TestPutNextToIdentical', 'TestUnblockingLoop']

level_list = [name for name, level in level_dict.items()]
# print(level_list)
# ['BossLevel', 'BossLevelNoUnlock', 
# 'GoTo', 'GoToImpUnlock', 
# 'GoToLocal', 'GoToLocalS5N2', 'GoToLocalS6N2', 'GoToLocalS6N3', 
# 'GoToLocalS6N4', 'GoToLocalS7N4', 'GoToLocalS7N5', 'GoToLocalS8N2', 
# 'GoToLocalS8N3', 'GoToLocalS8N4', 'GoToLocalS8N5', 'GoToLocalS8N6', 
# 'GoToLocalS8N7', 'GoToObj', 'GoToObjMaze', 'GoToObjMazeOpen', 
# 'GoToObjMazeS4', 'GoToObjMazeS4R2', 'GoToObjMazeS5', 'GoToObjMazeS6', 
# 'GoToObjMazeS7', 'GoToObjS4', 'GoToObjS6', 'GoToOpen', 'GoToRedBall',
#  'GoToRedBallGrey', 'GoToRedBallNoDists', 'GoToSeq', 'GoToSeqS5R2', 

# 'Open', 
# 
# 'Pickup', 'PickupLoc', 
# 
# 'PutNext', 'PutNextLocal', 'PutNextLocalS5N3', 'PutNextLocalS6N4', 
#
#  'Synth', 'SynthLoc', 'SynthS5R2', 'SynthSeq', 
# 
# 'UnblockPickup', 'Unlock', 

# 'TestGoToBlocked', 'TestLotsOfBlockers', 'TestPutNextCloseToDoor', 
# 'TestPutNextToBlocked', 'TestPutNextToCloseToDoor1', 'TestPutNextToCloseToDoor2', 
# 'TestPutNextToIdentical', 'TestUnblockingLoop']

"""
additional
"""
# 'MiniBossLevel', 
# '1RoomS12', '1RoomS16', '1RoomS20', '1RoomS8', 'ActionObjDoor', 'BlockedUnlockPickup', 
# 'FindObjS5', 'FindObjS6', 'FindObjS7', 
# 'GoToDoor', 'GoToObjDoor', 'GoToRedBlueBall', 
# 'KeyCorridorS3R1', 'KeyCorridorS3R2', 'KeyCorridorS3R3', 'KeyCorridorS4R3', 'KeyCorridorS5R3', 'KeyCorridorS6R3', 'KeyInBox', 
# 'MoveTwoAcrossS5N2', 'MoveTwoAcrossS8N9', 
# 'OpenDoor', 'OpenDoorColor', 'OpenDoorDebug', 'OpenDoorLoc', 'OpenDoorsOrderN2', 'OpenDoorsOrderN2Debug', 
# 'OpenDoorsOrderN4', 'OpenDoorsOrderN4Debug', 'OpenRedBlueDoors', 'OpenRedBlueDoorsDebug', 
# 'OpenRedDoor', 'OpenTwoDoors', 'OpenTwoDoorsDebug', 'PickupAbove', 'PickupDist', 'PickupDistDebug', 'PutNextS4N1',
#  'PutNextS5N1', 'PutNextS5N2', 'PutNextS5N2Carrying', 'PutNextS6N3', 'PutNextS6N3Carrying', 'PutNextS7N4', 
# 'PutNextS7N4Carrying', 'UnlockLocal', 'UnlockLocalDist', 'UnlockPickup', 'UnlockPickupDist', 'UnlockToUnlock', 
# 

import babyai
import gym


# level = 'BabyAI-GoToRedBall-v0' # go to a red ball
# level = 'BabyAI-OpenDoorLoc-v0' # open a door with location
# level = 'BabyAI-GoToObj-v0' # go to where
# level = 'BabyAI-ActionObjDoor-v0' # goto or open or pick up obj
# level = 'BabyAI-FindObjS6-v0' # pick up (ball,box,key)
# level = 'BabyAI-PutNext-v0' # put ... next to ...
# level = 'BabyAI-Open-v0' # == 'BabyAI-Unlock-v0'  # python train_rl.py --env 'BabyAI-Open-v0' --model ppo
# level = 'BabyAI-OpenDoorLoc-v0'
# level = 'BabyAI-OpenDoor-v0'
# level_ = 'BabyAI-GoToDoor-v0' # go to door
# level = 'BabyAI-GoTo-v0' # go to door or obj # python train_rl.py --env 'BabyAI-GoTo-v0' --model ppo
# level = 'BabyAI-Pickup-v0' # # python train_rl.py --env 'BabyAI-Pickup-v0' --model ppo

# level = 'BabyAI-PutNext-v0' # # python train_rl.py --env 'BabyAI-PutNext-v0' --model ppo
# level = 'BabyAI-PutNextS5N1-v0' # # python train_rl.py --env 'BabyAI-PutNext-v0' --model ppo
# level = 'BabyAI-PutNextS6N3-v0' # # python train_rl.py --env 'BabyAI-PutNext-v0' --model ppo
# level = 'BabyAI-PutNextS7N4-v0' # # python train_rl.py --env 'BabyAI-PutNext-v0' --model ppo
# level = 'BabyAI-PutNextS4N1-v0' # # python train_rl.py --env 'BabyAI-PutNext-v0' --model ppo
level = 'BabyAI-BossLevel-v0'
# level = 'BabyAI-Synth-v0'
n_episodes = 1000

env = gym.make(level)
instructions = set(env.reset()['mission'] for i in range(n_episodes))
print(len(instructions))
for instr in sorted(instructions):
    with open("/data2/username_high/username/BABYAI/logs/exp-4/static_instr.txt","a") as file:
        file.write(instr)
    

# for level in level_list:
#     if 'Test' in level:
#         continue
#     level_ = 'BabyAI-' + str(level) + '-v0'
#     # print(level_)
#     env = gym.make(level_)
#     instructions = set(env.reset()['mission'] for i in range(n_episodes))
#     if len(instructions) > 10:
#         print("-----------------------------------------")
#         print(level)
#         for instr in sorted(instructions):
#             print(instr)
#         print("-----------------------------------------")
