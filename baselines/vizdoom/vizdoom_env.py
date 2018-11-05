import vizdoom as vzd
import skimage.transform
import numpy as np

from gym.spaces import Discrete, Box

class ViZDoomENV:
    def __init__(self, seed, render=False, use_depth=False, use_rgb=True, reward_scale=1, frame_repeat=1, reward_reshape=False):
        # assign observation space
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        channel_num = 0
        if use_depth:
            channel_num = channel_num + 1
        if use_rgb:
            channel_num = channel_num + 3
        
        #self.observation_space = ViZDoom_observation_space((channel_num, 84, 84))
        self.observation_space = Box(low=0.0, high=1.0, shape=(84, 84, channel_num))

        self.reward_reshape = reward_reshape
        self.reward_scale = reward_scale
        
        
        game = vzd.DoomGame()
        game.set_doom_scenario_path("vizdoom/ViZDoom_map/health_gathering_supreme.wad")
        game.add_available_game_variable(vzd.GameVariable.HEALTH)
        
        # game input setup
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        if use_depth:
            game.set_depth_buffer_enabled(True)
        
        # rendering setup
        game.set_render_hud(False)
        game.set_render_minimal_hud(False)  # If hud is enabled
        game.set_render_crosshair(False)
        game.set_render_weapon(False)
        game.set_render_decals(False)  # Bullet holes and blood on the walls
        game.set_render_particles(False)
        game.set_render_effects_sprites(False)  # Smoke and blood
        game.set_render_messages(False)  # In-game messages
        game.set_render_corpses(False)
        #game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items
        
        # Adds buttons that will be allowed.
        self.action_space = Discrete(3)
        game.add_available_button(vzd.Button.TURN_LEFT)
        game.add_available_button(vzd.Button.TURN_RIGHT)
        game.add_available_button(vzd.Button.MOVE_FORWARD)
        # generate the corresponding actions
        num_buttons = self.action_space.n
        actions = [([False] * num_buttons)for i in range(num_buttons)]
        for i in range(num_buttons):
            actions[i][i] = True
        self.actions = actions
        # set frame repeat for taking action
        self.frame_repeat = frame_repeat
        
        # Causes episodes to finish after 2100 tics (actions)
        game.set_episode_timeout(2100)
        # Sets the livin reward (for each move) to 1
        game.set_living_reward(1)
        #game.set_death_penalty(1000 * reward_scale)
        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        game.set_mode(vzd.Mode.PLAYER)

        
        
        
        game.set_seed(seed)
        game.set_window_visible(render)
        game.init()
        
        self.game = game
                
        
    def get_current_input(self):
        state = self.game.get_state()

        resolution = self.observation_space.shape[:2]
        
        n = state.number
        
        if self.use_rgb:
            screen_buf = state.screen_buffer
            screen_buf = skimage.transform.resize(screen_buf, resolution)
            #screen_buf = np.rollaxis(screen_buf, 2, 0)
            res = screen_buf
        if self.use_depth:
            depth_buf = state.depth_buffer
            depth_buf = skimage.transform.resize(depth_buf, resolution)
            depth_buf = depth_buf[np.newaxis,:]
            res = depth_buf

        if self.use_depth and self.use_rgb:
            res = np.vstack((screen_buf, depth_buf))
        
        self.last_input = (res, n)
        
        return res, n
    
    def step(self, action):
        info = {}
        
        reward = self.game.make_action(self.actions[action], self.frame_repeat)
        if self.reward_reshape:
            fixed_shaping_reward = self.game.get_game_variable(vzd.GameVariable.USER1) 
            shaping_reward = vzd.doom_fixed_to_double(fixed_shaping_reward) 
            shaping_reward = shaping_reward - self.last_total_shaping_reward
            self.last_total_shaping_reward += shaping_reward
            reward = shaping_reward
        
        done = self.game.is_episode_finished()
        if done:
            ob, n = self.last_input
            info = {'episode':{'r':self.total_reward, 'l':n}}
            # info['Episode_Total_Reward'] = self.total_reward
            # info['Episode_Total_Len'] = n
        else:
            ob, n = self.get_current_input()
        
        reward = reward * self.reward_scale
        self.total_reward += reward


        return ob, reward, done, info
    
    def reset(self):
        self.last_input = None
        self.game.new_episode()
        self.last_total_shaping_reward = 0
        self.total_reward = 0
        ob, n = self.get_current_input()
        return ob
    
    def close(self):
        self.game.close()
