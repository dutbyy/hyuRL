from tests.local.trainer import main
from tests.local.entry import flow_config
from hyurl.tools.common import Summary, fix_print


if __name__ == "__main__":
    Summary.setpath('cartpole-v1-env')
    fix_print()
    main(flow_config)
