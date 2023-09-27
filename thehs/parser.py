import sys


class FLAGS:
    def __init__(self, parse_object):
        for key, value in parse_object.items():
            setattr(self, key.replace("-", "_"), value["value"])

    def __repr__(self):
        return str(self.__dict__)


class Parser:
    def __init__(self):
        self.__vars = {}

    def DEFINE_integer(self, name: str, default: int, hint: str = ""):
        self.__vars[name] = {"type": int, "value": default, "hint": hint}

    def DEFINE_float(self, name: str, default: float, hint: str = ""):
        self.__vars[name] = {"type": float, "value": default, "hint": hint}

    def DEFINE_string(self, name: str, default: str | None, hint: str = ""):
        self.__vars[name] = {"type": str, "value": default, "hint": hint}

    def DEFINE_bool(self, name: str, default: bool, hint: str = ""):
        self.__vars[name] = {"type": bool, "value": default, "hint": hint}

    def __match(self):
        argvs = sys.argv[1:]
        vars = self.__vars.keys()

        if "-h" in argvs:
            if argvs[0] != "-h":
                raise ValueError("""Help flag ['-h'] must be at first.""")
            else:
                if len(argvs) == 1:
                    print("Usage:")
                    for k, v in self.__vars.items():
                        print(f" - Argument: {k}, type: {v['type']}")
                        print(f"            {v['hint']}")
                else:
                    for arg in argvs[1:]:
                        n_arg = arg[2:]
                        v = self.__vars[n_arg]
                        print(f" - Argument: {n_arg}, type: {v['type']}")
                        print(f"            {v['hint']}")
            exit(0)
        for argv in argvs:
            arg, val = argv.strip().split("=")
            if arg.startswith("--"):
                arg = arg[2:]
                if arg in vars:
                    var_info = self.__vars[arg]
                    var_info["value"] = var_info["type"](val)
                    var_type = var_info["type"]
                    if var_type is int:
                        var_info["value"] = int(val)
                    elif var_type is float:
                        var_info["value"] = int(val)
                    elif var_type is str:
                        var_info["value"] = val
                    elif var_type is bool:
                        if val == "True":
                            var_info["value"] = True
                        elif val == "False":
                            var_info["value"] = False
                        else:
                            ValueError("Value mismatch with variable's type!")
                    else:
                        raise ValueError("Argument's type does not supported!")
                else:
                    raise ValueError(f"Argument {arg} does not exist!")
            else:
                raise ValueError("Invalid flag prefix!")

    def parse(self, fetch=True):
        if fetch:
            self.__match()
        return FLAGS(self.__vars)