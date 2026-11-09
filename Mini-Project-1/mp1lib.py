import numpy as np


def bsc(bits, p, rng):
    """Pass bits through a binary symmetric channel.
    Each bit is independently flipped with probability p."""
    
    bits = np.asarray(bits, dtype=int)
    flips = rng.random(bits.shape) < p
    return np.bitwise_xor(bits, flips.astype(int))

def packet_errors(n_pkt, n_bits, p, rng):
    """Return the number of errors in each packet."""
    return rng.binomial(n=n_bits, p=p, size=n_pkt)



def attempts_until_clean(n_trials, q, rng):
    
    #Generate the number of transmission attempts needed to obtain
    #a clean packet.

    #q is the clean-packet probability for one attempt.
    #The returned geometric random variable begins at 1.
    
    return rng.geometric(p=q, size=n_trials)


def tv_distance(pmf1, pmf2):
    
    #Calculate total variation distance between two probability
    #mass functions defined on the same support.
    
    pmf1 = np.asarray(pmf1, dtype=float)
    pmf2 = np.asarray(pmf2, dtype=float)

    if pmf1.shape != pmf2.shape:
        raise ValueError("The two PMFs must have the same shape.")

    return 0.5 * np.sum(np.abs(pmf1 - pmf2))