import os


def fix_print():
    import builtins
    import datetime
    import inspect

    origin_print = builtins.print

    def custom_print(*args, **kwargs):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        prefix = f"[{timestamp}] [{os.path.basename(caller.filename)}:{caller.lineno}]"
        origin_print(prefix, *args, **kwargs)

    builtins.print = custom_print


def exmain():
    from hyuRL.src.tools.infer import inference, init_env, init_model
    from hyuRL.example.atari.entry import flow_config

    # model_path_dict = {"cp_model": "/job/model/cp_model/cp_model_200.pth"}
    model_path_dict = {}
    flow_env = init_env(flow_config=flow_config, env_id=1)
    name2model = init_model(flow_config=flow_config, model_path_dict=model_path_dict)
    inference(flow_env, name2model, episode_num=10000)


if __name__ == "__main__":
    fix_print()
    exmain()
