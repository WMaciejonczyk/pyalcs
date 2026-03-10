import logging
import random
import numpy as np
from lcs import Perception
from lcs.agents.Agent import TrialMetrics
from lcs.agents.acs2eder.ReplayMemory import ReplayMemory
from lcs.agents.acs2eder.ReplayMemorySample import ReplayMemorySample
from lcs.strategies.action_selection.BestAction import BestAction
from lcs.agents.acs2 import ClassifiersList
from lcs.agents.acs2 import Configuration
from lcs.agents.Agent import Agent

logger = logging.getLogger(__name__)


class ACS2EDER(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList = None) -> None:
        self.cfg = cfg
        self.population = population or ClassifiersList()
        self.replay_memory = ReplayMemory(max_size=cfg.eder_buffer_size)
        self.diversity_scores = []

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, time, current_trial=None) \
            -> TrialMetrics:

        logger.debug("** Running trial explore ** ")
        # Initial conditions
        steps = 0
        state = env.reset()
        last_reward = 0
        action = env.action_space.sample()
        prev_state = Perception.empty()
        action_set = ClassifiersList()
        done = False

        trajectory = []

        while not done:
            state_p = Perception(state)
            assert len(state) == self.cfg.classifier_length

            match_set = self.population.form_match_set(state)

            if steps > 0:
                ClassifiersList.apply_alp(
                    self.population, match_set, action_set, prev_state,
                    action, state_p, time + steps, self.cfg.theta_exp,
                    self.cfg)

                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, match_set.get_maximum_fitness(),
                    self.cfg.beta, self.cfg.gamma)

                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps, self.population, match_set, action_set,
                        state_p, self.cfg.theta_ga, self.cfg.mu, self.cfg.chi,
                        self.cfg.theta_as, self.cfg.do_subsumption,
                        self.cfg.theta_exp)

            action = self.cfg.action_selector(match_set)
            action_set = match_set.form_action_set(action)

            prev_state = Perception(state_p)
            raw_state, last_reward, done, _ = env.step(action)
            state = Perception(raw_state)

            sample = ReplayMemorySample(
                prev_state, action, last_reward, state, done
            )
            trajectory.append(sample)

            if done:
                ClassifiersList.apply_alp(
                    self.population, ClassifiersList(), action_set, prev_state,
                    action, state, time + steps, self.cfg.theta_exp,
                    self.cfg)

                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta, self.cfg.gamma)

                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps, self.population, ClassifiersList(),
                        action_set,
                        state, self.cfg.theta_ga, self.cfg.mu,
                        self.cfg.chi,
                        self.cfg.theta_as, self.cfg.do_subsumption,
                        self.cfg.theta_exp)

            state = raw_state
            steps += 1

        segments = self._segment_trajectory(trajectory)
        scores = self._compute_diversity_score(segments)

        M = max(scores)

        for segment, Qj in zip(segments, scores):

            alpha = Qj / M if M > 0 else 0.0
            u = random.random()

            if u <= alpha:
                self.replay_memory.update((segment, Qj))

        segments = self._sample_segments_prioritized(
            self.cfg.eder_samples_number
        )

        for segment in segments:
            for sample in segment:
                eder_match_set = self.population.form_match_set(
                    sample.state)
                eder_action_set = eder_match_set.form_action_set(
                    sample.action)
                eder_next_match_set = self.population.form_match_set(
                    sample.next_state)
                # Apply learning in the replied action set
                ClassifiersList.apply_alp(
                    self.population,
                    eder_next_match_set,
                    eder_action_set,
                    sample.state,
                    sample.action,
                    sample.next_state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                ClassifiersList.apply_reinforcement_learning(
                    eder_action_set,
                    sample.reward,
                    0 if sample.done else eder_next_match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma
                )
                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps,
                        self.population,
                        ClassifiersList() if sample.done else eder_next_match_set,
                        eder_action_set,
                        sample.next_state,
                        self.cfg.theta_ga,
                        self.cfg.mu,
                        self.cfg.chi,
                        self.cfg.theta_as,
                        self.cfg.do_subsumption,
                        self.cfg.theta_exp)

        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(self, env, time=None, current_trial=None) \
            -> TrialMetrics:

        logger.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0
        state = Perception(env.reset())

        last_reward = 0
        action_set = ClassifiersList()
        done = False

        while not done:
            match_set = self.population.form_match_set(state)

            if steps > 0:
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma)

            # Here when exploiting always choose best action
            action = BestAction(
                all_actions=self.cfg.number_of_possible_actions)(match_set)
            action_set = match_set.form_action_set(action)

            state, last_reward, done, _ = env.step(action)
            state = Perception(state)

            if done:
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta, self.cfg.gamma)

            steps += 1

        return TrialMetrics(steps, last_reward)

    def _segment_trajectory(self, trajectory):
        segments = []
        b = self.cfg.eder_subtrajectory_length
        if len(trajectory) <= b:
            segments.append(trajectory)
        else:
            for i in range(0, len(trajectory) - b + 1, b):
                segments.append(trajectory[i:i + b])

        return segments

    def _compute_diversity_score(self, segments):
        scores = []

        for segment in segments:
            states = [np.array(s.state, dtype=float) for s in segment]
            M = np.stack(states, axis=1)

            norms = np.linalg.norm(M, axis=0, keepdims=True) + 1e-10
            M = M / norms
            L = np.dot(M.T, M)
            try:
                Lc = np.linalg.cholesky(L+ np.eye(len(L)) * 1e-6)
                score = np.prod(np.square(np.diagonal(Lc)))
            except np.linalg.LinAlgError:
                score = 1e-9
            scores.append(score)

        return scores

    def _sample_segments_prioritized(self, n):
        if len(self.replay_memory) == 0:
            return []

        # segments = []
        # scores = []
        #
        # for seg, q in self.replay_memory:
        #     segments.append(seg)
        #     scores.append(q)
        segments, scores = zip(*self.replay_memory)
        if len(self.replay_memory) < n:
            return segments
        else:
            scores = np.array(scores, dtype=float)

            probs = scores / np.sum(scores)

            indices = np.random.choice(
                len(segments),
                size=n,
                replace=False,
                p=probs
            )

            return [segments[i] for i in indices]
