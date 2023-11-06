# NOTE: Signature v2 Evolved
from Levenshtein import distance

class SignaturesEvolved:

    def __init__(
            self,
            signatures,
            true_positive_threshold=6,
            sliding_window_stride=1,
            sliding_window_padding=2
    ):
        self.signatures = signatures
        self.true_positive_threshold = true_positive_threshold
        self.sliding_window_stride = sliding_window_stride
        self.sliding_window_padding = sliding_window_padding

    
    def sliding_window_similarity(self, signature, command):
        window_size = len(signature) + self.sliding_window_padding * 2 # padding from both sides
        min_dist = float('inf')
        best_match = None
        
        for i in range(len(command) - window_size + self.sliding_window_stride):
            window = command[i:i+window_size]
            dist = distance(signature, window)
            if dist < min_dist:
                min_dist = dist
                best_match = window
                
        return min_dist, best_match
            
    def analyze_command(self, command):
        """
        This function takes a command and makes an analysis whether a True Positive should be triggered.

        Args:
            command (str): command to be analyzed
            true_positive_treshold (float): treshold for true positive
            stride (int): stride for sliding window
        
        Returns:
            bool: True if true positive should be triggered, False otherwise
            distance (int): distance between signature and command
            signature (str): signature that was matched
            fragment (str): fragment of command that matched signature
        """
        for signature in self.signatures:
            distance, fragment = self.sliding_window_similarity(signature, command)
            if distance <= self.true_positive_threshold:
                return True, distance, signature, fragment
        return False, None, None, None  # If no signature is close enough

    def estimate_distance_threshold(self, baseline, fpr_max=1e-4):
        """
        TODO: It is possible to estimate distance threshold automatically given some set of baseline is known.
        This method might iterate over baseline, collect min disances and return the distance that 
        corresponds to the given false positive rate.
        """
        raise NotImplementedError


class SignaturesEvolvedAugmented(SignaturesEvolved):
    """
    In addition to SignatureEvolved expands list of singatures
    by performing an augmentation on signatures to address variability.
    """
    def __init__(self):
        pass