import madrona_python
import gpu_hideseek_python
import torch
import torchvision
import sys
import termios
import tty

def get_single_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())
    
    ch = sys.stdin.read(1)
    
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return ch

class Action:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.r = 0
        self.g = 0
        self.l = 0
        self.reset = 0

def get_keyboard_action():
    while True:
        key_action = get_single_char()

        result = Action()

        if key_action == 'w':
            result.y = 3
        elif key_action == 'a':
            result.x = -3
        elif key_action == 'd':
            result.x = 3
        elif key_action == 's':
            result.y = -3
        elif key_action == 'q':
            result.r = 1
        elif key_action == 'e':
            result.r = -1
        elif key_action == 'g':
            result.g = 1
        elif key_action == 'l':
            result.l = 1
        elif key_action == 'x':
            result.reset = -1
        elif key_action == '1':
            result.reset = 1
        elif key_action == '2':
            result.reset = 2
        elif key_action == '3':
            result.reset = 3
        elif key_action == '4':
            result.reset = 4
        elif key_action == '5':
            result.reset = 5
        elif key_action == ' ':
            pass
        else:
            continue

        return result

sim = gpu_hideseek_python.HideAndSeekSimulator(
        exec_mode = gpu_hideseek_python.ExecMode.CUDA,
        gpu_id = 0,
        num_worlds = 1,
        min_entities_per_world = 5,
        max_entities_per_world = 5,
        render_width = 1024,
        render_height = 1024,
        debug_compile = False,
)

actions = sim.action_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()
agent_valid_masks = sim.agent_mask_tensor().to_torch()
agent_visibility_masks = sim.visibility_masks_tensor().to_torch()
resets = sim.reset_tensor().to_torch()
rgb_observations = sim.rgb_tensor().to_torch()
print(actions.shape, actions.dtype)
print(resets.shape, resets.dtype)
print(rgb_observations.shape, rgb_observations.dtype)

while True:
    print("Stepping")
    sim.step()
    torchvision.utils.save_image((rgb_observations[0].float() / 255).permute(2, 0, 1), sys.argv[1])

    print(rewards * agent_valid_masks)
    print(agent_visibility_masks * agent_valid_masks)

    action = get_keyboard_action()
    
    if action.reset < 0:
        break

    resets[0] = action.reset
    actions[0][0][0] = action.x
    actions[0][0][1] = action.y
    actions[0][0][2] = action.r
    actions[0][0][3] = action.g
    actions[0][0][4] = action.l

del sim
