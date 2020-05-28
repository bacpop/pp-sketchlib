#!/usr/bin/env python
# Copyright 2020 John Lees

# Adapted from Common Substrings in Random Strings Blais & Blanchette 2006
# See Equation 4

from math import factorial

# recursive version
def bernoulli_enum(alphabet, k, seq = "", patterns = []):
    if len(seq) < k:
        for idx, letter in enumerate(alphabet):
           bernoulli_enum(alphabet[idx:], k, seq + letter, patterns)
    else:
        patterns.append(seq)

    return(patterns)

def prob_word(enum, alphabet, pvec):
    prob_pi = 1
    for letter, prob in zip(alphabet, pvec):
        prob_pi *= prob ** enum.count(letter)
    return(prob_pi)

def p_gamma(enum, alphabet, pvec, rc, length):
    prob_pi = prob_word(enum, alphabet, pvec)
    return 1 - (1 - rc*prob_pi)**(length - len(enum) + 1)

def omega(enum, alphabet):
    denominator = 1
    for letter in alphabet:
        denominator *= factorial(enum.count(letter))
    return(factorial(len(enum))/denominator)

# pwr 2 here is hard coded r, number of seqs
def match_prob_csrs(alphabet, k, pvec, rc, length):
    prob_pi = 1
    for enum in bernoulli_enum(alphabet, k):
        prob_pi *= (1 - p_gamma(enum, alphabet, pvec, rc, length)**2)**omega(enum, alphabet)
    return 1 - prob_pi

def expected_match(alphabet, k, pvec_query, pvec_reference, rc, length):
    mean_sum = 0
    weight_sum = 0
    for enum in bernoulli_enum(alphabet, k):
        count = omega(enum, alphabet)
        word_p = prob_word(enum, alphabet, pvec_query)
        weight_sum += count**2 * word_p
        mean_sum += p_gamma(enum, alphabet, pvec_reference, rc, length) * word_p * count**2
    return mean_sum/(weight_sum)

def jaccard_expected(e1, e2):
    return((e1 * e2) / (e1 + e2 - e1*e2))


alphabet = ['A', 'C', 'G', 'T']
k = 11
pvec1 = [0.25, 0.25, 0.25, 0.25]
pvec2 = [0.05, 0.45, 0.45, 0.05]
rc = 1
length = 50000

e1 = expected_match(alphabet, k, pvec1, pvec2, rc, length)

e2 = expected_match(alphabet, k, pvec2, pvec1, rc, length)

print(e1)
print(e2)
print(jaccard_expected(e1, e2))


e11 = expected_match(alphabet, k, pvec1, pvec1, rc, length)
e22 = expected_match(alphabet, k, pvec2, pvec2, rc, length)

print(e11)
print(jaccard_expected(e11, e11))
print(e22)
print(jaccard_expected(e22, e22))

print(match_prob_csrs(alphabet, 6, [0.2962, 0.2037, 0.2035, 0.2966], 1, 18))
