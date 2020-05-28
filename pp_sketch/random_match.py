#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees

from math import factorial

# recursive version
def bernoulli_enum(alphabet, k, seq = "", patterns = []):
    if len(seq) < k:
        for idx, letter in enumerate(alphabet):
           bernoulli_enum(alphabet[idx:], k, seq + letter, patterns)
    else:
        patterns.append(seq)

    return(patterns)

def p_gamma(enum, alphabet, pvec, rc, length):
    prob_pi = 1
    for letter, prob in zip(alphabet, pvec):
        prob_pi *= prob ** enum.count(letter)
    return 1 - (1 - rc*prob_pi)**(length - len(enum) - 1)

def omega(enum, alphabet):
    denominator = 1
    for letter in alphabet:
        denominator *= factorial(enum.count(letter))
    return(factorial(len(enum))/denominator)

def match_prob_csrs(alphabet, k, pvec, rc, length):
    prob_pi = 1
    for enum in bernoulli_enum(alphabet, k):
        print(p_gamma(enum, alphabet, pvec, rc, length))
        print((1 - p_gamma(enum, alphabet, pvec, rc, length)**2)**omega(enum, alphabet))
        prob_pi *= (1 - p_gamma(enum, alphabet, pvec, rc, length)**2)**omega(enum, alphabet)
    return 1 - prob_pi

def expected_match(alphabet, k, pvec, rc, length):
    count_sum = 0
    mean_sum = 0
    for enum in bernoulli_enum(alphabet, k):
        count = omega(enum, alphabet)
        count_sum += count
        mean_sum += p_gamma(enum, alphabet, pvec, rc, length) * count
    return mean_sum/count_sum

def jaccard_expected(e1, e2):
    return((e1 * e2) / (e1 + e2 - e1*e2))


alphabet = ['A', 'C', 'G', 'T']
k = 9
pvec = [0.25, 0.25, 0.25, 0.25]
rc = 2
length = 50000

e1 = expected_match(alphabet, k, pvec, rc, length)

pvec = [0.05, 0.45, 0.45, 0.05]

e2 = expected_match(alphabet, k, pvec, rc, length)

print(jaccard_expected(e1, e2))

##
#
# Think we may want E(p_gamma) (seems to be the case for equal pvec)
# i.e. 1/(sum(omega)) * {p_gamma[0] * omega[0] + p_gamma[1] * omega[1] + ...}
#
#
#

