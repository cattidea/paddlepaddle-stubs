from __future__ import annotations

import paddle
from paddle.sysconfig import get_include, get_lib


def main():
    print(get_lib())
    print(paddle.sysconfig.get_lib())
    # paddle.utils.run_check()


if __name__ == "__main__":
    main()
