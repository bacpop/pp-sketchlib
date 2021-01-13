#ifndef STHASH__ITERATOR_H
#define STHASH__ITERATOR_H 1

#include <string>
#include <limits>
#include "nthash.hpp"

/**
 * Iterate over hash values for k-mers in a
 * given DNA sequence.
 *
 * This implementation uses ntHash
 * function to efficiently calculate
 * hash values for successive k-mers.
 */

class stHashIterator
{

public:
  static std::vector<std::vector<unsigned>> parseSeed(const std::vector<std::string> &seedString)
  {
    std::vector<std::vector<unsigned>> seedSet;
    for (unsigned i = 0; i < seedString.size(); i++)
    {
      std::vector<unsigned> sSeed;
      for (unsigned j = 0; j < seedString[i].size(); j++)
      {
        if (seedString[i][j] != '1')
          sSeed.push_back(j);
      }
      seedSet.push_back(sSeed);
    }
    return seedSet;
  }

  /**
     * Default constructor. Creates an iterator pointing to
     * the end of the iterator range.
    */
  stHashIterator() : m_hVec(NULL),
                     m_hStn(NULL),
                     m_pos(std::numeric_limits<std::size_t>::max())
  {
  }

  /**
     * Constructor.
     * @param seq address of DNA sequence to be hashed
     * @param seed address of spaced seed
     * @param k k-mer size
     * @param h number of seeds
     * @param h2 number of hashes per seed
     * @param rc use canonical k-mers (allow reverse complement)
     * @param ss use spaced seeds
    */
  stHashIterator(const std::string &seq, const std::vector<std::vector<unsigned>> &seed, unsigned h, unsigned h2, unsigned k, bool rc, bool ss) : m_seq(seq), m_seed(seed), m_h(h), m_h2(h2), m_k(k), m_rc(rc), m_ss(ss),
                                                                                                                                                  m_hVec(new uint64_t[h * h2]), m_minhVec(new uint64_t[h2]), m_hStn(new bool[h * h2]), m_pos(0)
  {
    init();
  }

  /** Initialize internal state of iterator */
  void init()
  {
    if (m_k > m_seq.length())
    {
      m_pos = std::numeric_limits<std::size_t>::max();
      return;
    }
    unsigned locN = 0;
    if (m_ss)
    {
      while (m_pos < m_seq.length() - m_k + 1 && (m_rc ? !NTMSMC64(m_seq.data() + m_pos, m_seed, m_k, m_h, m_h2, m_fhVal, m_rhVal, locN, m_hVec, m_minhVec, m_hStn)
                                                       : !NTMSM64(m_seq.data() + m_pos, m_seed, m_k, m_h, m_h2, m_fhVal, locN, m_hVec, m_minhVec)))
        m_pos += locN + 1;
    }
    else
    {
      while (m_pos < m_seq.length() - m_k + 1 && (m_rc ? !NTMC64(m_seq.data() + m_pos, m_k, m_h2, m_fhVal, m_rhVal, locN, m_hVec)
                                                       : !NTM64(m_seq.data() + m_pos, m_k, m_h2, m_fhVal, locN, m_hVec)))
        m_pos += locN + 1;
    }

    if (m_pos >= m_seq.length() - m_k + 1)
      m_pos = std::numeric_limits<std::size_t>::max();
  }

  /** Advance iterator right to the next valid k-mer */
  void next()
  {
    ++m_pos;
    if (m_pos >= m_seq.length() - m_k + 1)
    {
      m_pos = std::numeric_limits<std::size_t>::max();
      return;
    }
    if (seedTab[(unsigned char)(m_seq.at(m_pos + m_k - 1))] == seedN)
    {
      m_pos += m_k;
      init();
    }
    else
    {
      if (m_rc && m_ss)
      {
        NTMSMC64(m_seq.data() + m_pos, m_seed, m_seq.at(m_pos - 1), m_seq.at(m_pos - 1 + m_k), m_k, m_h, m_h2, m_fhVal, m_rhVal, m_hVec, m_minhVec, m_hStn);
      }
      else if (m_ss)
      {
        NTMSM64(m_seq.data() + m_pos, m_seed, m_seq.at(m_pos - 1), m_seq.at(m_pos - 1 + m_k), m_k, m_h, m_h2, m_fhVal, m_hVec, m_minhVec);
      }
      else if (m_rc)
      {
        NTMC64(m_seq.at(m_pos - 1), m_seq.at(m_pos - 1 + m_k), m_k, m_h, m_fhVal, m_rhVal, m_hVec);
      }
      else
      {
        NTM64(m_seq.at(m_pos - 1), m_seq.at(m_pos - 1 + m_k), m_k, m_h, m_hVec);
      }
    }
  }

  size_t pos() const
  {
    return m_pos;
  }

  /** get pointer to strand for current k-mer */
  const bool *strandArray() const
  {
    return m_hStn;
  }

  /** get pointer to hash values for current k-mer */
  const uint64_t *operator*() const
  {
    if (m_ss)
    {
      return m_minhVec;
    }
    else
    {
      return m_hVec;
    }
  }

  /** test equality with another iterator */
  bool operator==(const stHashIterator &it) const
  {
    return m_pos == it.m_pos;
  }

  /** test inequality with another iterator */
  bool operator!=(const stHashIterator &it) const
  {
    return !(*this == it);
  }

  /** pre-increment operator */
  stHashIterator &operator++()
  {
    next();
    return *this;
  }

  /** iterator pointing to one past last element */
  static const stHashIterator end()
  {
    return stHashIterator();
  }

  /** destructor */
  ~stHashIterator()
  {
    if (m_hVec != NULL)
    {
      delete[] m_hVec;
      delete[] m_minhVec;
      delete[] m_hStn;
    }
  }

private:
  /** DNA sequence */
  std::string m_seq;

  /** Spaced Seed sequence */
  std::vector<std::vector<unsigned>> m_seed;

  /** number of seeds */
  unsigned m_h;

  /** number of hashes per seed */
  unsigned m_h2;

  /** k-mer size */
  unsigned m_k;

  /** canonical k-mers */
  bool m_rc;

  /** spaced seeds */
  bool m_ss;

  /** hash values
     *  For m_h = n and m_h2 = m:
     *  [seed1Hash1, seed1Hash2 ... seed(n)Hash(m-1), seed(n)Hash(m)]
     *  minhVec is the minimum across all seeds
    */
  uint64_t *m_hVec;
  uint64_t *m_minhVec;

  /** hash strands, forward = 0, reverse-complement = 1 */
  bool *m_hStn;

  /** position of current k-mer */
  size_t m_pos;

  /** forward-strand k-mer hash value */
  uint64_t m_fhVal;

  /** reverse-complement k-mer hash value */
  uint64_t m_rhVal;
};

#endif
