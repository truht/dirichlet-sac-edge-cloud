# A Stabilized Dirichlet Soft Actor-Critic Controller for AI Workload-Aware Offloading across the Edge-Cloud Continuum

Reference implementation accompanying the paper submitted to *Future Generation
Computer Systems* (Elsevier), Virtual Special Issue "Lightweight and
Sustainable Computing across the Cloud-Edge Continuum".

This repository is released independently of the review outcome, as stated in
the manuscript's Data Availability and Conclusion sections.

## Contents

- `drl_controller/` — the Dirichlet-parameterized SAC+TD3 routing controller
  (Python/PyTorch). Implements the fixed-temperature objective, the
  concentration dual-ascent penalty, and the TD3 clamp-and-renormalize
  stabilization described in Section 4 of the paper.
  - `sac/agent.py` — actor/critic update loop, target-policy smoothing,
    delayed actor updates (`policy_delay`), dual-ascent concentration penalty.
  - `sac/networks.py` — actor (Dirichlet concentration head) and twin critic
    networks.
  - `sac/replay_buffer.py` — off-policy transition buffer.
  - `reward.py` — CVaR / Jain's-fairness / drop-penalty reward components.
  - `config.py` — the single fixed hyperparameter set used across all five
    device-count scales (300-2000) reported in the paper.
  - `train.py`, `protocol.py` — training loop entry point and the
    JSON wire protocol used to talk to the simulator (see below).

- `edgecloudsim-ai/` — the simulation environment, built as a modification of
  [EdgeCloudSim](https://github.com/CagataySonmez/EdgeCloudSim)
  (Sonmez et al.) with:
  - `src/edu/boun/edgecloudsim/drl/` — the socket bridge, DRL-aware
    orchestrator and mobile-device manager that let the external Python
    controller drive routing decisions at each decision step.
  - `src/edu/boun/edgecloudsim/gpu_dual/` — the dual CPU/GPU AI-workload
    model (`t_eff = max(cpuTime, gpuTime)`), reused unmodified from the
    authors' prior conference paper (ATC 2026) as Section 3.1 of this
    manuscript.
  - `src/edu/boun/edgecloudsim/applications/drl_app/` and
    `scripts/drl_app/` — the application wiring and run scripts used for
    every experiment reported in the paper.
  - `lib/` — third-party build dependencies bundled by upstream EdgeCloudSim
    (CloudSim, Commons Math, Colt, Weka, jFuzzyLogic).

- `figures/` — scripts that regenerate the paper's result figures directly
  from the verified numbers in the manuscript's tables (e.g. `make_pareto.py`
  regenerates the Pareto scatter figure from Table 4).

## What is NOT included here

Raw experiment logs and the offline RL transition dataset are not included in
this code repository (they are large binary/log data, not code). They are
released separately; see the Data Availability section of the manuscript for
the corresponding link.

## Build and run

The Java side is an Eclipse project (`.classpath`/`.project` included) that
depends on the JARs in `edgecloudsim-ai/lib/`. For the general EdgeCloudSim
build/run workflow (Eclipse import, `compile.sh`/`runner.sh` conventions),
see the upstream project's documentation; `scripts/drl_app/compile.sh` and
`scripts/drl_app/runner.sh` follow the same convention for this application.

The Python side requires PyTorch; `drl_controller/train.py` is the entry
point and `protocol.py` documents the JSON message format exchanged with the
Java simulator over the socket bridge.

## Citation

If you use this code, please cite the FGCS paper (details to be added upon
publication) and the prior conference paper this environment and AI-workload
model originate from:

> Modeling AI Workloads and Evaluating Their Impact on Edge Task Offloading
> Algorithms, International Conference on Advanced Technologies for
> Communications (ATC 2026).

and the original simulator:

> C. Sonmez, A. Ozgovde, C. Ersoy, "EdgeCloudSim: An environment for
> performance evaluation of edge computing systems," Transactions on Emerging
> Telecommunications Technologies, 2018.

## License

The `edgecloudsim-ai/` directory is a derivative work of EdgeCloudSim and is
licensed under the **GNU General Public License v3.0** (see `LICENSE`), as
required by the upstream project's license. The `drl_controller/` and
`figures/` directories are original contributions of this paper's authors,
also made available under GPLv3 for consistency with the rest of this
repository.
