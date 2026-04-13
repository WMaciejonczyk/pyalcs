import lcs.agents.acs2 as acs2


class Configuration(acs2.Configuration):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

        # EDER replay memory buffer size
        self.eder_buffer_size = kwargs.get('eder_buffer_size', 5000)

        # EDER replay memory min samples
        self.eder_min_samples = kwargs.get('eder_min_samples', 1000)

        # EDER replay memory samples number
        self.eder_samples_number = kwargs.get('eder_samples_number', 5)

        # EDER subtrajectory length
        self.eder_subtrajectory_length = kwargs.get(
            'eder_subtrajectory_length', 4)

    def __str__(self) -> str:
        return str(vars(self))
