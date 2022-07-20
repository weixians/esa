from crowd_nav.first_step import initialize
from pf_helper import agent_builder, pf_runner, network_builder

if __name__ == "__main__":
    args = initialize(default_policy_name="esa", default_random_seed=42, use_pfrl=True)
    q_func = network_builder.build_q_fuc(args)
    agent = agent_builder.build_dqn_agent(args, q_func)
    if not args.test:
        pf_runner.train(args, agent)
    else:
        # 加载模型
        agent.load(args.model_dir)
        pf_runner.eval_agent(args, agent, "test", 0)
