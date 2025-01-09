import os
import logging


def fix_print():
    import builtins
    origin_print = builtins.print
    def custom_print(*args, **kwargs):
        import datetime
        import inspect
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        prefix = f"[{timestamp}] [{os.path.basename(caller.filename)}:{caller.lineno}]"
        origin_print(prefix, *args, **kwargs)
    builtins.print = custom_print

def exmain():
    # import drill
    # drill.local_run = True
    from hyuRL.src.tools.infer import inference, init_env, init_model
    from hyuRL.example.minigame.entry import flow_config
    model_path_dict = {}
    flow_env = init_env(flow_config=flow_config, env_id=1)
    name2model = init_model(flow_config=flow_config, model_path_dict=model_path_dict)
    inference(flow_env, name2model, episode_num=10000)

if __name__ == '__main__':
    fix_print()
    exmain()
