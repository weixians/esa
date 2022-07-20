import json
import logging
import os
from multiprocessing import Pool

from crowd_nav import global_util
from crowd_nav.first_step import initialize
from pf_helper import agent_builder, pf_runner, network_builder

logger = global_util.setup_logger(use_file_log=False)
n_episodes = 1000
test_time_limit = 25


def run_test(choice):
    policy_name, seed = choice
    dynamic_human_nums = [5, 10, 15, 20]
    static_human_nums = [1, 2, 3, 4, 5]

    output_dir = os.path.join(global_util.get_project_root(), "../output/0720_42")
    model_subdir = "epi_10000"

    dynamic_results = {}
    for dynamic_num in dynamic_human_nums:
        args = initialize(
            default_policy_name=policy_name,
            default_random_seed=seed,
            default_dynamic_num=dynamic_num,
            default_static_num=0,
            test_time_limit=test_time_limit,
            default_output_dir=output_dir,
            default_test=True,
            use_pfrl=True,
            logger=logger,
        )
        args.logger = logger
        args.writer = None
        q_func = network_builder.build_q_fuc(args)
        agent = agent_builder.build_dqn_agent(args, q_func)
        agent.load(os.path.join(args.model_dir, model_subdir))
        r = pf_runner.eval_agent(args, agent, "test", 0, n_episodes=n_episodes)
        dynamic_results[str(dynamic_num)] = r

    static_results = {}
    for static_num in static_human_nums:
        args = initialize(
            default_policy_name=policy_name,
            default_random_seed=seed,
            default_dynamic_num=5,
            default_static_num=static_num,
            test_time_limit=test_time_limit,
            default_output_dir=output_dir,
            default_test=True,
            use_pfrl=True,
            logger=logger,
        )
        q_func = network_builder.build_q_fuc(args)
        agent = agent_builder.build_dqn_agent(args, q_func)
        agent.load(os.path.join(args.model_dir, model_subdir))
        r = pf_runner.eval_agent(args, agent, "test", 0, n_episodes=n_episodes)
        static_results[str(static_num)] = r

    results = {"dynamic": dynamic_results, "static": static_results}

    test_result_dir_path = os.path.join(global_util.get_project_root(), "../test_result")
    os.makedirs(test_result_dir_path, exist_ok=True)
    with open(
        os.path.join(test_result_dir_path, "{}_seed{}.json".format(policy_name, seed)),
        "w",
    ) as f:
        json.dump(results, f)

    logging.info("Policy: {}, seed={}, 评估结束".format(policy_name, seed))

    return results


if __name__ == "__main__":
    policy_names = ["sarl", "lstm_rl", "esa"]
    # policy_names = ["sarl"]
    seeds = [42]

    choices = []
    for name in policy_names:
        for seed in seeds:
            choices.append((name, seed))
            # run_test((name, seed))

    with Pool(3) as p:
        print(p.map(run_test, choices))
        p.close()
