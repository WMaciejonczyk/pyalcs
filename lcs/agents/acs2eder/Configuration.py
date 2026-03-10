import lcs.agents.acs2 as acs2


class Configuration(acs2.Configuration):
    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

        # EDER replay memory buffer size
        self.eder_buffer_size = kwargs.get('eder_buffer_size', 10000)

        # EDER replay memory samples number
        self.eder_samples_number = kwargs.get('eder_samples_number', 8)

        self.eder_subtrajectory_length = kwargs.get(
            'eder_subtrajectory_length', 10)

    def __str__(self) -> str:
        return str(vars(self))
