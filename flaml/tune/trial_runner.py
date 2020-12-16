from ray.tune.trial import Trial
from ray.tune.schedulers.trial_scheduler import TrialScheduler

class BaseTrialRunner:
    """Implementation of a sequential trial runner

    Note that the caller usually should not mutate trial state directly.
    """

    def __init__(self, 
        search_alg=None, 
        scheduler=None, 
        metric: str = None,
        mode: str = 'min',):
        self._search_alg = search_alg
        self._scheduler_alg = scheduler 
        self._trials = []
        self._metric = metric
        self._mode = mode
    
    def get_trials(self):
        """Returns the list of trials managed by this TrialRunner.

        Note that the caller usually should not mutate trial state directly.
        """
        return self._trials

    def add_trial(self, trial):
        """Adds a new trial to this TrialRunner.

        Trials may be added at any time.

        Args:
            trial (Trial): Trial to queue.
        """
        self._trials.append(trial)
        self._scheduler_alg.on_trial_add(self, trial)

    def process_trial_result(self, trial, result):
        self._search_alg.on_trial_result(trial.trial_id, result)
        decision = self._scheduler_alg.on_trial_result(self, trial, result)
        if decision == TrialScheduler.STOP: trial.status = Trial.TERMINATED
        if decision == TrialScheduler.PAUSE: trial.status = Trial.PAUSED 


    def stop_trial(self, trial):
        """Stops trial.
        - Original doc from ray tune's trial runner:
        Trials may be stopped at any time. If trial is in state PENDING
        or PAUSED, calls `on_trial_remove`  for scheduler and
        `on_trial_complete() for search_alg.
        Otherwise waits for result for the trial and calls
        `on_trial_complete` for scheduler and search_alg if RUNNING.
        
        - Our revision:
        If status is not ERROR or TERMINATED, call
        self._scheduler_alg.on_trial_remove(self, trial) and
        self._search_alg.on_trial_complete.
        """
        if trial.status in [Trial.ERROR, Trial.TERMINATED]:
            return
        else:
            self._scheduler_alg.on_trial_remove(self, trial)
            self._search_alg.on_trial_complete(trial.trial_id)


class SequentialTrialRunner(BaseTrialRunner):
    """Implementation of the sequential trial runner

    .. test in pseudocode-block: 

        from ray.tune.suggest import SearchGenerator
        from ray.tune.schedulers import FIFOScheduler
        search_algs = SearchGenerator
        scheduler_alg = FIFOScheduler
        runner = TrialRunner(
            search_alg=search_alg,
            scheduler_alg=FIFOScheduler,
            metric='loss',
            mode='min',
            )
        while in_budget:
            trial_to_run = runner.step()
            result = trainable_func(trial_to_run)
            runner.process_trial_result(trial_to_run, result)
    """

    def step(self) -> Trial:
        """Runs one step of the trial event loop.
        Callers should typically run this method repeatedly in a loop. They
        may inspect or modify the runner's state in between calls to step().

        returns a Trial to run
        """
        #TODO: only showing a simple example. need to re-write.
        trial = self._search_alg.next_trial()
        if trial: self.add_trial(trial)
        trial_to_run = self._scheduler_alg.choose_trial_to_run(self)
        trial_to_run.status = Trial.RUNNING
        return trial_to_run
