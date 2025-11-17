def get_policy(name, config):
    if name == "heuristic":
        from .heuristic_policy_dna import HeuristicPolicy
        return HeuristicPolicy(config)
    # elif name == "dnn":
    #     from .dnn_policy import DNNPolicy
    #     return DNNPolicy(config)
    # elif name == "reinforce learning":
    #     from .rl_policy import ReinforcePolicy
    #     return ReinforcePolicy(config)
    if name == "profile":
        from .profile_policy import ProfilePolicy
        return ProfilePolicy(config)