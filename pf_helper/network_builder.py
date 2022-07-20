from models.esa_model import EsaModel
from models.sarl_model import SarlModel
from models.lstm_rl_model import LstmRlModel
from pf_helper import pf_runner


def build_q_fuc(args):
    input_dim = 12
    args.recurrent = False
    if args.network == "esa":
        q_func = EsaModel(
            config=args.policy_config, n_actions=args.env.action_space_size, input_dim=input_dim, device=args.device
        )
    elif args.network == "sarl":
        q_func = SarlModel(
            config=args.policy_config, n_actions=args.env.action_space_size, input_dim=input_dim, device=args.device
        )
    elif args.network == "lstm_rl":
        q_func = LstmRlModel(
            config=args.policy_config, n_actions=args.env.action_space_size, input_dim=input_dim, device=args.device
        )
    else:
        raise NotImplementedError("No network found by name: {}".format(args.network))

    if args.load_pretrain:
        pf_runner.load_pretrained_weights(args.device, "{}.pth".format(args.network), q_func)

    # torchsummaryX.summary(q_func, torch.zeros([1, 5, 12]))

    return q_func
