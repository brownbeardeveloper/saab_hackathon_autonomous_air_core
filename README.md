![Frontend](frontend.png)

# Autonomous Air Core - Autonomous Airbase Management

**Description**

Autonomous Air Core is an autonomous airbase management system that uses reinforcement learning to optimize aircraft maintenance and transfer decisions. The system is designed to operate in a simulated environment where an AI agent controls a fleet of aircraft and makes decisions about maintenance, transfers, refueling, and mission assignments to maximize mission success and minimize downtime.

This project is part of the SAAB Hackathon 2026, where participants were challenged to design a smart airbase. Our team had previously developed another component of a separate project that simulates an airbase and its aircraft for educational purposes. This repository focuses on reinforcement learning, where an agent is trained to manage the airbase and coordinate aircraft operations.

I did not have enough time to deploy a fully trained model for the hackathon, so we had to use a "stupid" trained model that could not make optimal decisions. It would sometimes refuel an aircraft even when it already had 100% fuel, which was pretty funny. We maybe should add penalty when the agent refuels an aircraft that already had more than 80% fuel.

The project does not have a very clean folder structure because it was built quickly, like a real hackathon project.

The hackathon was fun and we learned a lot.

-----

**Technical Details**

- The RL agent is trained using the Stable Baselines3 library and the PPO algorithm.
- Training takes place in a simulated airbase environment where the agent makes decisions about maintenance aircraft transfers, refueling, and mission assignments.
- The agent is first trained using a hyperparameter tuning sweep.
- The final model should then be trained again using the best hyperparameter configuration found during the sweep.
- Due to time constraints, this final training step was not completed, so the best configuration from the sweep was used directly.
- The agent uses a masked action space, since some actions are not valid in certain states of the environment.
- This prevents the agent from selecting impossible actions and improves training efficiency.

**Penalty rules**
- If no aircraft are ready for a mission, the agent receives a penalty.
- If the airbase does not have enough fuel for refueling, the agent receives a penalty.
- If the airbase does not have enough weapons available for a loadout change, the agent receives a penalty.

----

## Training

Run a hyperparameter tuning sweep with:

```bash
python sweep_train.py --tuning
```

Run the final training after tuning with:

```bash
python sweep_train.py --final
```

`--final` uses `final_training.selected_profile` from `training_profiles.yml` when it is set. If it is empty or `best`, it reuses the best hyperparameter configuration from the latest tuning summary in `artifacts/`.

----

## Quick Start for Demo

* Run `cd frontend && npm i && npm run dev` in the terminal.

----

## Discord notifications

Discord notifications are optional. When enabled with `DISCORD_WEBHOOK_URL`:

- benchmark sweeps send one message each time a profile finishes
- final training sends a progress message every 10%
- final training sends a completion message when it finishes
- training sends a failure message if the run crashes

----

## Todo

- [ ] Adjust the reward function so the agent learns to make better decisions
- [ ] Adjust the configuration for the hyperparameter sweep and start training the agents
- [ ] Run the final training with the best hyperparameter configuration
- [ ] Evaluate the results in jupyter notebook