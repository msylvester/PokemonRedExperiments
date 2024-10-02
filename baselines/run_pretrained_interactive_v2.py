from os.path import exists
from pathlib import Path
import uuid
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

# Session path and model file definitions
sessionPath = '../sesh/session_c0b039c0'
#initial_model_file = 'session_4da05e87_main_good/poke_439746560_steps.zip'  # Initial model file path
initial_model_file = '../first_v2_session/poke_1310720_steps.zip'  # Initial model file path
model_paths = {
    'first': 'poke_1310720_steps.zip',
    'second': 'poke_1638400_steps.zip',
    'third': 'poke_327680_steps.zip',
    'fourth': 'poke_655360_steps.zip',
    'fifth': 'poke_983040_steps.zip'
}

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)  # Default environment
        return env
    set_random_seed(seed)
    return _init
def save_state(env, file_path="saved_state.state"):
    with open(file_path, 'wb') as f:
        env.pyboy.save_state(f)
    print(f"State saved to {file_path}")

def switch_to_new_env_and_model(env_config, model_file):
    """Switches to a new environment and loads a new model."""

    # Create a new dictionary with only the supported keys
    # supported_env_config = {
    #     'init_state': env_config['init_state'],
    #     'max_steps': env_config['max_steps'],
    #     'gb_path': env_config['gb_path'],
    #     'save_final_state': env_config['save_final_state'],
    #     'early_stop': env_config['early_stop'],
    #     'print_rewards': env_config['print_rewards'],
    #     'save_video': env_config['save_video'],
    #     'fast_video': env_config['fast_video'],
    #     'debug': env_config['debug'],
    #     'extra_buttons': env_config['extra_buttons']
    #     # Omit 'headless' and 'action_frequency'
    # }
    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': True, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
    }

    env = RedGymEnv(env_config)  # Unpack the supported_env_config dictionary

    # Check if the model file exists
    if not exists(model_file):
        print(f"Model file {model_file} not found.")
        exit()

    print('\nLoading checkpoint for new environment from:', model_file)
    model = PPO.load(model_file, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    return env, model

if __name__ == '__main__':
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23

    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../state_step_9095.state', 'max_steps': ep_length, 
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': True, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': True
    }

    num_cpu = 1  # Number of CPUs/environments to run
    env = make_env(0, env_config)()  # Single environment setup

    # Load the initial model
    print('\nLoading initial checkpoint')
    model = PPO.load(initial_model_file, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
    
    obs, info = env.reset()
    max_steps = 100000  # Set a limit for the number of steps to prevent infinite loops
    step_count = 0  # Initialize step counter
    model_switched = False  # Boolean flag to track model switch

    #for saving 
    save_interval = 2000  # Save every 1000 steps (for example)
    step_count = 0
    while True:
        step_count += 1  # Increment step counter

        # Check if agent is enabled based on agent_enabled.txt
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].strip().lower() == "yes"
        except FileNotFoundError:
            agent_enabled = False

        # If agent is enabled, predict action using the model
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        else:
            # If agent is disabled, set a safe default action
            action = 7  # Replace with a safe default action for your environment

        # Step through the environment with the chosen action
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()

        badges = env.get_badges()
        print(f"The badges are {badges}")
        # if(badges == 1):
        #     save_state(env, f"state_step_{step_cou
        # nt}.state")

        # Check for model switch after 200 steps
        '''
        we dont need to switch the model but uncomment if you want to
        '''
        # if step_count >= 50 and not model_switched:
        #     print("Switching to new environment and model...")
        #     env, model = switch_to_new_env_and_model(env_config, f'../sesh/session_c0b039c0/{model_paths["fifth"]}')  # Pass the fifth model file here
        #     model_switched = True  # Set the flag to True after switching

        # Print game status from the info dictionary
        location = info.get('location', 'Unknown')
        health = info.get('player_health', 'Unknown')
        enemies_defeated = info.get('enemies_defeated', 0)

        print(f"Enemies Defeated: {enemies_defeated}")
        step_count += 1
        if step_count % save_interval == 0:
            save_state(env, f"state_step_{step_count}.state")

        if truncated:
            break
        # Check for termination conditions or step limit
        if step_count >= max_steps or terminated or truncated:
            print(f"Exiting loop after {step_count} steps.")
            break

    env.close()
