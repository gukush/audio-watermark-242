
from abc import ABC, abstractmethod

class BaseCloner(ABC):
    name = "base"
    @abstractmethod
    def extract_voice(self,audio):
        """
        For extracting voice from audio sample to later use for cloning.
        To be implemented by derived class.
        """
        pass

    @abstractmethod
    def apply_voice(self,input, output):
        """
        For transforming voice from one sample to previously extracted one.
        To be implemented by derived class.
        """
        pass
    @abstractmethod
    def clone_voice_to_sample(self,sample,voice,output):
        """
        For transforming voice from one sample to previously extracted one.
        To be implemented by derived class.
        """
        pass
