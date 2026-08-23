import torch


def save_random_state():

    state = {"cpu": torch.get_rng_state()}

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()

    return state


def restore_random_state(state):

    torch.set_rng_state(state["cpu"])

    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])
