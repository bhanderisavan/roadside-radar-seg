from datetime import datetime


class TimeStamp:
    """
    This class defines the data structure to store one Epoch time in raw and human readable processed format
    """

    def __init__(self, epoch_time_upto_milliseconds):

        date_time = datetime.utcfromtimestamp(epoch_time_upto_milliseconds / 1000)

        self.year = date_time.year
        self.month = date_time.month
        self.day = date_time.day
        self.hour = date_time.hour
        self.minute = date_time.minute
        self.second = date_time.second
        self.milliseconds = int(date_time.microsecond / 1000)
        self.epoch_value = int(epoch_time_upto_milliseconds)

    def __sub__(self, other):
        """
        This is method overloading for subtracting two timestamp objects and return the result in seconds.
        """
        date_time_1 = datetime.fromtimestamp(self.epoch_value / 1000)
        data_time_2 = datetime.fromtimestamp(other.epoch_value / 1000)

        diff = date_time_1 - data_time_2
        return diff.total_seconds()

    def __repr__(self):
        # to print the timestamp in human readable format
        human_readable_time = (
            "%.4d" % (self.year)
            + "-"
            + "%.2d" % (self.month)
            + "-"
            + "%.2d" % (self.day)
            + " "
            + "%.2d" % (self.hour)
            + ":"
            + "%.2d" % (self.minute)
            + ":"
            + "%.2d" % (self.second)
            + "."
            + "%.3d" % (self.milliseconds)
        )

        out = (
            "["
            + human_readable_time
            + "] "
            + "epoch = ["
            + str(self.epoch_value)
            + "]\n"
        )
        return out
