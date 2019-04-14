class LearningSchedule:
    """

    """
    def __init__(self, schedule):
        self.schedule = dict(schedule)

    def compute_learning_parameter_value(self, iter_index):
        """

        :param iter_index:
        :return:
        """
        sorted_terminal_iters = list(sorted(self.get_schedule().keys()))
        for terminal_iter in sorted_terminal_iters:
            if terminal_iter >= iter_index:
                return self.get_schedule()[terminal_iter]
        return self.get_schedule()[sorted_terminal_iters.pop()]

    def get_schedule(self):
        """

        :return:
        """
        return self.schedule

    def set_schedule(self, schedule):
        """

        :param schedule:
        :return:
        """
        return \
            LearningSchedule(
                schedule=schedule
            )
