from abc import ABC, abstractmethod

class BaseMetric(ABC):

    @abstractmethod
    def evaluate_quality(self, audio):
        """
        Method to evaluate the quality of the audio.
        Should be implemented by any derived class.
        """
        pass
