import os
from crowd_nav import global_util
from crowd_nav.first_step import initialize
from pf_helper import network_builder, agent_builder, pf_runner

logger = global_util.setup_logger(use_file_log=False)
test_time_limit = 25

if __name__ == "__main__":
    policy_names = ["sarl", "lstm_rl", "esa"]
    output_dir = os.path.join(global_util.get_project_root(), "../output/final")
    model_subdir = "epi_10000"

    dynamic_human_nums = [5, 10, 15, 20]
    static_human_nums = [1, 2, 3, 4, 5]
    seed = 0

    for policy_name in policy_names:
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
            q_func = network_builder.build_q_fuc(args)
            agent = agent_builder.build_dqn_agent(args, q_func)
            agent.load(os.path.join(args.model_dir, model_subdir))
            for i in range(20):
                pf_runner.visualize_test(args, agent, i)
