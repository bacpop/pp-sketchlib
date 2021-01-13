/*
 *
 * nthash.hpp
 * Author: Hamid Mohamadi
 * Genome Sciences Centre,
 * British Columbia Cancer Agency
 *
 * Modified by John Lees (2020)
 */
#pragma once

#include <vector>

#include "nthash_tables.hpp"
#include "bitfuncs.hpp"

// rotate "v" to the left 1 position
inline uint64_t rol1(const uint64_t v)
{
  return (v << 1) | (v >> 63);
}

// rotate "v" to the right by 1 position
inline uint64_t ror1(const uint64_t v)
{
  return (v >> 1) | (v << 63);
}

// rotate 31-left bits of "v" to the left by "s" positions
inline uint64_t rol31(const uint64_t v, unsigned s)
{
  s %= 31;
  return ((v << s) | (v >> (31 - s))) & 0x7FFFFFFF;
}

// rotate 33-right bits of "v" to the left by "s" positions
inline uint64_t rol33(const uint64_t v, unsigned s)
{
  s %= 33;
  return ((v << s) | (v >> (33 - s))) & 0x1FFFFFFFF;
}

// swap bit 0 with bit 33 in "v"
inline uint64_t swapbits033(const uint64_t v)
{
  uint64_t x = (v ^ (v >> 33)) & 1;
  return v ^ (x | (x << 33));
}

// swap bit 32 with bit 63 in "v"
inline uint64_t swapbits3263(const uint64_t v)
{
  uint64_t x = ((v >> 32) ^ (v >> 63)) & 1;
  return v ^ ((x << 32) | (x << 63));
}

// forward-strand hash value of the base kmer, i.e. fhval(kmer_0)
inline uint64_t NTF64(const char *kmerSeq, const unsigned k)
{
  uint64_t hVal = 0;
  for (unsigned i = 0; i < k; i++)
  {
    hVal = rol1(hVal);
    hVal = swapbits033(hVal);
    hVal ^= seedTab[(unsigned char)kmerSeq[i]];
  }
  return hVal;
}

// reverse-strand hash value of the base kmer, i.e. rhval(kmer_0)
inline uint64_t NTR64(const char *kmerSeq, const unsigned k)
{
  uint64_t hVal = 0;
  for (unsigned i = 0; i < k; i++)
  {
    hVal = rol1(hVal);
    hVal = swapbits033(hVal);
    hVal ^= seedTab[(unsigned char)kmerSeq[k - 1 - i] & cpOff];
  }
  return hVal;
}

// forward-strand ntHash for sliding k-mers
inline uint64_t NTF64(const uint64_t fhVal, const unsigned k, const unsigned char charOut, const unsigned char charIn)
{
  uint64_t hVal = rol1(fhVal);
  hVal = swapbits033(hVal);
  hVal ^= seedTab[charIn];
  hVal ^= (msTab31l[charOut][k % 31] | msTab33r[charOut][k % 33]);
  return hVal;
}

// reverse-complement ntHash for sliding k-mers
inline uint64_t NTR64(const uint64_t rhVal, const unsigned k, const unsigned char charOut, const unsigned char charIn)
{
  uint64_t hVal = rhVal ^ (msTab31l[charIn & cpOff][k % 31] | msTab33r[charIn & cpOff][k % 33]);
  hVal ^= seedTab[charOut & cpOff];
  hVal = ror1(hVal);
  hVal = swapbits3263(hVal);
  return hVal;
}

// canonical ntBase
inline uint64_t NTC64(const char *kmerSeq, const unsigned k)
{
  uint64_t fhVal = 0, rhVal = 0;
  fhVal = NTF64(kmerSeq, k);
  rhVal = NTR64(kmerSeq, k);
  return (rhVal < fhVal) ? rhVal : fhVal;
}

// canonical ntHash
inline uint64_t NTC64(const char *kmerSeq, const unsigned k, uint64_t &fhVal, uint64_t &rhVal)
{
  fhVal = NTF64(kmerSeq, k);
  rhVal = NTR64(kmerSeq, k);
  return (rhVal < fhVal) ? rhVal : fhVal;
}

// canonical ntHash for sliding k-mers
inline uint64_t NTC64(const unsigned char charOut, const unsigned char charIn, const unsigned k, uint64_t &fhVal, uint64_t &rhVal)
{
  fhVal = NTF64(fhVal, k, charOut, charIn);
  rhVal = NTR64(rhVal, k, charOut, charIn);
  return (rhVal < fhVal) ? rhVal : fhVal;
}

// forward-strand ntHash for sliding k-mers to the left
inline uint64_t NTF64L(const uint64_t rhVal, const unsigned k, const unsigned char charOut, const unsigned char charIn)
{
  uint64_t hVal = rhVal ^ (msTab31l[charIn][k % 31] | msTab33r[charIn][k % 33]);
  hVal ^= seedTab[charOut];
  hVal = ror1(hVal);
  hVal = swapbits3263(hVal);
  return hVal;
}

// reverse-complement ntHash for sliding k-mers to the left
inline uint64_t NTR64L(const uint64_t fhVal, const unsigned k, const unsigned char charOut, const unsigned char charIn)
{
  uint64_t hVal = rol1(fhVal);
  hVal = swapbits033(hVal);
  hVal ^= seedTab[charIn & cpOff];
  hVal ^= (msTab31l[charOut & cpOff][k % 31] | msTab33r[charOut & cpOff][k % 33]);
  return hVal;
}

// canonical ntHash for sliding k-mers to the left
inline uint64_t NTC64L(const unsigned char charOut, const unsigned char charIn, const unsigned k, uint64_t &fhVal, uint64_t &rhVal)
{
  fhVal = NTF64L(fhVal, k, charOut, charIn);
  rhVal = NTR64L(rhVal, k, charOut, charIn);
  return (rhVal < fhVal) ? rhVal : fhVal;
}

// ntBase with seeding option
inline uint64_t NTF64(const char *kmerSeq, const unsigned k, const unsigned seed)
{
  uint64_t hVal = NTF64(kmerSeq, k);
  if (seed == 0)
    return hVal;
  hVal *= seed ^ k * multiSeed;
  hVal ^= hVal >> multiShift;
  return hVal;
}

// canonical ntBase with seeding option
inline uint64_t NTC64(const char *kmerSeq, const unsigned k, const unsigned seed)
{
  uint64_t hVal = NTC64(kmerSeq, k);
  if (seed == 0)
    return hVal;
  hVal *= seed ^ k * multiSeed;
  hVal ^= hVal >> multiShift;
  return hVal;
}

// multihash ntHash, ntBase
inline void NTM64(const char *kmerSeq, const unsigned k, const unsigned m, uint64_t *hVal)
{
  uint64_t bVal = 0, tVal = 0;
  bVal = NTF64(kmerSeq, k);
  hVal[0] = bVal;
  for (unsigned i = 1; i < m; i++)
  {
    tVal = bVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    hVal[i] = tVal;
  }
}

// one extra hash for given base hash
inline uint64_t NTE64(const uint64_t hVal, const unsigned k, const unsigned i)
{
  uint64_t tVal = hVal;
  tVal *= (i ^ k * multiSeed);
  tVal ^= tVal >> multiShift;
  return tVal;
}

// multihash ntHash for sliding k-mers
inline void NTM64(const unsigned char charOut, const unsigned char charIn, const unsigned k, const unsigned m, uint64_t *hVal)
{
  uint64_t bVal = 0, tVal = 0;
  bVal = NTF64(hVal[0], k, charOut, charIn);
  hVal[0] = bVal;
  for (unsigned i = 1; i < m; i++)
  {
    tVal = bVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    hVal[i] = tVal;
  }
}

// canonical multihash ntBase
inline void NTMC64(const char *kmerSeq, const unsigned k, const unsigned m, uint64_t *hVal)
{
  uint64_t bVal = 0, tVal = 0;
  bVal = NTC64(kmerSeq, k);
  hVal[0] = bVal;
  for (unsigned i = 1; i < m; i++)
  {
    tVal = bVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    hVal[i] = tVal;
  }
}

// canonical multihash ntHash
inline void NTMC64(const char *kmerSeq, const unsigned k, const unsigned m, uint64_t &fhVal, uint64_t &rhVal, uint64_t *hVal)
{
  uint64_t bVal = 0, tVal = 0;
  bVal = NTC64(kmerSeq, k, fhVal, rhVal);
  hVal[0] = bVal;
  for (unsigned i = 1; i < m; i++)
  {
    tVal = bVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    hVal[i] = tVal;
  }
}

// canonical multihash ntHash for sliding k-mers
inline void NTMC64(const unsigned char charOut, const unsigned char charIn, const unsigned k, const unsigned m, uint64_t &fhVal, uint64_t &rhVal, uint64_t *hVal)
{
  uint64_t bVal = 0, tVal = 0;
  bVal = NTC64(charOut, charIn, k, fhVal, rhVal);
  hVal[0] = bVal;
  for (unsigned i = 1; i < m; i++)
  {
    tVal = bVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    hVal[i] = tVal;
  }
}

/*
 * ignoring k-mers containing nonACGT using ntHash function
*/

// canonical ntBase
inline bool NTC64(const char *kmerSeq, const unsigned k, uint64_t &hVal, unsigned &locN)
{
  hVal = 0;
  locN = 0;
  uint64_t fhVal = 0, rhVal = 0;
  for (int i = k - 1; i >= 0; i--)
  {
    if (seedTab[(unsigned char)kmerSeq[i]] == seedN)
    {
      locN = i;
      return false;
    }
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= seedTab[(unsigned char)kmerSeq[k - 1 - i]];

    rhVal = rol1(rhVal);
    rhVal = swapbits033(rhVal);
    rhVal ^= seedTab[(unsigned char)kmerSeq[i] & cpOff];
  }
  hVal = (rhVal < fhVal) ? rhVal : fhVal;
  return true;
}

// canonical multihash ntBase
inline bool NTMC64(const char *kmerSeq, const unsigned k, const unsigned m, unsigned &locN, uint64_t *hVal)
{
  uint64_t bVal = 0, tVal = 0, fhVal = 0, rhVal = 0;
  locN = 0;
  for (int i = k - 1; i >= 0; i--)
  {
    if (seedTab[(unsigned char)kmerSeq[i]] == seedN)
    {
      locN = i;
      return false;
    }
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= seedTab[(unsigned char)kmerSeq[k - 1 - i]];

    rhVal = rol1(rhVal);
    rhVal = swapbits033(rhVal);
    rhVal ^= seedTab[(unsigned char)kmerSeq[i] & cpOff];
  }
  bVal = (rhVal < fhVal) ? rhVal : fhVal;
  hVal[0] = bVal;
  for (unsigned i = 1; i < m; i++)
  {
    tVal = bVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    hVal[i] = tVal;
  }
  return true;
}

// canonical ntHash
inline bool NTC64(const char *kmerSeq, const unsigned k, uint64_t &fhVal, uint64_t &rhVal, uint64_t &hVal, unsigned &locN)
{
  hVal = fhVal = rhVal = 0;
  locN = 0;
  for (int i = k - 1; i >= 0; i--)
  {
    if (seedTab[(unsigned char)kmerSeq[i]] == seedN)
    {
      locN = i;
      return false;
    }
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= seedTab[(unsigned char)kmerSeq[k - 1 - i]];

    rhVal = rol1(rhVal);
    rhVal = swapbits033(rhVal);
    rhVal ^= seedTab[(unsigned char)kmerSeq[i] & cpOff];
  }
  hVal = (rhVal < fhVal) ? rhVal : fhVal;
  return true;
}

// canonical multihash ntHash
inline bool NTMC64(const char *kmerSeq, const unsigned k, const unsigned m, uint64_t &fhVal, uint64_t &rhVal, unsigned &locN, uint64_t *hVal)
{
  fhVal = rhVal = 0;
  uint64_t bVal = 0, tVal = 0;
  locN = 0;
  for (int i = k - 1; i >= 0; i--)
  {
    if (seedTab[(unsigned char)kmerSeq[i]] == seedN)
    {
      locN = i;
      return false;
    }
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= seedTab[(unsigned char)kmerSeq[k - 1 - i]];

    rhVal = rol1(rhVal);
    rhVal = swapbits033(rhVal);
    rhVal ^= seedTab[(unsigned char)kmerSeq[i] & cpOff];
  }
  bVal = (rhVal < fhVal) ? rhVal : fhVal;
  hVal[0] = bVal;
  for (unsigned i = 1; i < m; i++)
  {
    tVal = bVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    hVal[i] = tVal;
  }
  return true;
}

// strand-preserving multihash ntHash
inline bool NTM64(const char *kmerSeq, const unsigned k, const unsigned m, uint64_t &fhVal, unsigned &locN, uint64_t *hVal)
{
  fhVal = 0;
  uint64_t bVal = 0, tVal = 0;
  locN = 0;
  for (int i = k - 1; i >= 0; i--)
  {
    if (seedTab[(unsigned char)kmerSeq[i]] == seedN)
    {
      locN = i;
      return false;
    }
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= seedTab[(unsigned char)kmerSeq[k - 1 - i]];
  }
  bVal = fhVal;
  hVal[0] = bVal;
  for (unsigned i = 1; i < m; i++)
  {
    tVal = bVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    hVal[i] = tVal;
  }
  return true;
}

// strand-aware canonical multihash ntHash
inline bool NTMC64(const char *kmerSeq, const unsigned k, const unsigned m, uint64_t &fhVal, uint64_t &rhVal, unsigned &locN, uint64_t *hVal, bool &hStn)
{
  fhVal = rhVal = 0;
  uint64_t bVal = 0, tVal = 0;
  locN = 0;
  for (int i = k - 1; i >= 0; i--)
  {
    if (seedTab[(unsigned char)kmerSeq[i]] == seedN)
    {
      locN = i;
      return false;
    }
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= seedTab[(unsigned char)kmerSeq[k - 1 - i]];

    rhVal = rol1(rhVal);
    rhVal = swapbits033(rhVal);
    rhVal ^= seedTab[(unsigned char)kmerSeq[i] & cpOff];
  }
  hStn = rhVal < fhVal;
  bVal = hStn ? rhVal : fhVal;
  hVal[0] = bVal;
  for (unsigned i = 1; i < m; i++)
  {
    tVal = bVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    hVal[i] = tVal;
  }
  return true;
}

// strand-aware canonical multihash ntHash for sliding k-mers
inline void NTMC64(const unsigned char charOut, const unsigned char charIn, const unsigned k, const unsigned m, uint64_t &fhVal, uint64_t &rhVal, uint64_t *hVal, bool &hStn)
{
  uint64_t bVal = 0, tVal = 0;
  bVal = NTC64(charOut, charIn, k, fhVal, rhVal);
  hStn = rhVal < fhVal;
  hVal[0] = bVal;
  for (unsigned i = 1; i < m; i++)
  {
    tVal = bVal * (i ^ k * multiSeed);
    tVal ^= tVal >> multiShift;
    hVal[i] = tVal;
  }
}

// masking canonical ntHash using spaced seed pattern
inline uint64_t maskHash(uint64_t &fkVal, uint64_t &rkVal, const char *seedSeq, const char *kmerSeq, const unsigned k)
{
  uint64_t fsVal = fkVal, rsVal = rkVal;
  for (unsigned i = 0; i < k; i++)
  {
    if (seedSeq[i] != '1')
    {
      fsVal ^= (msTab31l[(unsigned char)kmerSeq[i]][(k - 1 - i) % 31] | msTab33r[(unsigned char)kmerSeq[i]][(k - 1 - i) % 33]);
      rsVal ^= (msTab31l[(unsigned char)kmerSeq[i] & cpOff][i % 31] | msTab33r[(unsigned char)kmerSeq[i] & cpOff][i % 33]);
    }
  }
  return (rsVal < fsVal) ? rsVal : fsVal;
}

// spaced seed ntHash for base kmer, i.e. fhval(kmer_0)
inline uint64_t NTS64(const char *kmerSeq, const std::vector<bool> &seed, const unsigned k, uint64_t &hVal)
{
  hVal = 0;
  uint64_t sVal = 0;
  for (unsigned i = 0; i < k; i++)
  {
    hVal = rol1(hVal);
    hVal = swapbits033(hVal);
    sVal = hVal;
    hVal ^= seedTab[(unsigned char)kmerSeq[i]];
    if (seed[i])
      sVal = hVal;
  }
  return sVal;
}

// spaced seed ntHash for sliding k-mers
inline uint64_t NTS64(const char *kmerSeq, const std::vector<bool> &seed, const unsigned char charOut, const unsigned char charIn, const unsigned k, uint64_t &hVal)
{
  hVal = NTF64(hVal, k, charOut, charIn);
  uint64_t sVal = hVal;
  for (unsigned i = 0; i < k; i++)
    if (!seed[i])
    {
      sVal ^= (msTab31l[(unsigned char)kmerSeq[i]][k % 31] | msTab33r[(unsigned char)kmerSeq[i]][k % 33]);
    }
  return sVal;
}

// strand-aware multihash spaced seed ntHash
inline bool NTMS64(const char *kmerSeq, const std::vector<std::vector<unsigned>> &seedSeq, const unsigned k, const unsigned m, uint64_t &fhVal, uint64_t &rhVal, unsigned &locN, uint64_t *hVal, bool *hStn)
{
  fhVal = rhVal = 0;
  locN = 0;
  for (int i = k - 1; i >= 0; i--)
  {
    if (seedTab[(unsigned char)kmerSeq[i]] == seedN)
    {
      locN = i;
      return false;
    }
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= seedTab[(unsigned char)kmerSeq[k - 1 - i]];

    rhVal = rol1(rhVal);
    rhVal = swapbits033(rhVal);
    rhVal ^= seedTab[(unsigned char)kmerSeq[i] & cpOff];
  }

  for (unsigned j = 0; j < m; j++)
  {
    uint64_t fsVal = fhVal, rsVal = rhVal;
    for (unsigned i = 0; i < k; i++)
    {
      if (!seedSeq[j][i])
      {
        fsVal ^= (msTab31l[(unsigned char)kmerSeq[i]][(k - 1 - i) % 31] | msTab33r[(unsigned char)kmerSeq[i]][(k - 1 - i) % 33]);
        rsVal ^= (msTab31l[(unsigned char)kmerSeq[i] & cpOff][i % 31] | msTab33r[(unsigned char)kmerSeq[i] & cpOff][i % 33]);
      }
    }
    hStn[j] = rsVal < fsVal;
    hVal[j] = hStn[j] ? rsVal : fsVal;
  }
  return true;
}

// strand-aware multihash spaced seed ntHash for sliding k-mers
inline void NTMS64(const char *kmerSeq, const std::vector<std::vector<unsigned>> &seedSeq, const unsigned char charOut, const unsigned char charIn, const unsigned k, const unsigned m, uint64_t &fhVal, uint64_t &rhVal, uint64_t *hVal, bool *hStn)
{
  fhVal = NTF64(fhVal, k, charOut, charIn);
  rhVal = NTR64(rhVal, k, charOut, charIn);
  for (unsigned j = 0; j < m; j++)
  {
    uint64_t fsVal = fhVal, rsVal = rhVal;
    for (unsigned i = 0; i < k; i++)
    {
      if (!seedSeq[j][i])
      {
        fsVal ^= (msTab31l[(unsigned char)kmerSeq[i]][(k - 1 - i) % 31] | msTab33r[(unsigned char)kmerSeq[i]][(k - 1 - i) % 33]);
        rsVal ^= (msTab31l[(unsigned char)kmerSeq[i] & cpOff][i % 31] | msTab33r[(unsigned char)kmerSeq[i] & cpOff][i % 33]);
      }
    }
    hStn[j] = rsVal < fsVal;
    hVal[j] = hStn[j] ? rsVal : fsVal;
  }
}

// multihash spaced seed ntHash with multiple hashes per seed
inline bool NTMSM64(const char *kmerSeq, const std::vector<std::vector<unsigned>> &seedSeq, const unsigned k, const unsigned m, const unsigned m2,
                    uint64_t &fhVal, unsigned &locN, uint64_t *hVal, uint64_t *minhVal)
{
  fhVal = 0;
  locN = 0;
  for (int i = k - 1; i >= 0; i--)
  {
    if (seedTab[(unsigned char)kmerSeq[i]] == seedN)
    {
      locN = i;
      return false;
    }
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= seedTab[(unsigned char)kmerSeq[k - 1 - i]];
  }

  for (unsigned j = 0; j < m; j++)
  {
    uint64_t fsVal = fhVal;
    for (unsigned i = 0; i < k; i++)
    {
      if (!seedSeq[j][i])
      {
        fsVal ^= (msTab31l[(unsigned char)kmerSeq[i]][(k - 1 - i) % 31] | msTab33r[(unsigned char)kmerSeq[i]][(k - 1 - i) % 33]);
      }
    }
    hVal[j * m2] = fsVal;
    for (unsigned j2 = 1; j2 < m2; j2++)
    {
      uint64_t tVal = hVal[j * m2] * (j2 ^ k * multiSeed);
      tVal ^= tVal >> multiShift;
      hVal[j * m2 + j2] = tVal;
    }
  }
  for (unsigned int hIt = 0; hIt < m2; hIt++)
  {
    minhVal[hIt] = hVal[hIt * m];
    for (unsigned int seedIt = 1; seedIt < m; seedIt++)
    {
      minhVal[hIt] = MIN(minhVal[hIt], hVal[hIt * m + seedIt]);
    }
  }
  return true;
}

// multihash spaced seed ntHash for sliding k-mers with multiple hashes per seed
inline void NTMSM64(const char *kmerSeq, const std::vector<std::vector<unsigned>> &seedSeq, const unsigned char charOut, const unsigned char charIn,
                    const unsigned k, const unsigned m, const unsigned m2, uint64_t &fhVal, uint64_t *hVal, uint64_t *minhVal)
{
  fhVal = NTF64(fhVal, k, charOut, charIn);
  for (unsigned j = 0; j < m; j++)
  {
    uint64_t fsVal = fhVal;
    for (unsigned i = 0; i < k; i++)
    {
      if (!seedSeq[j][i])
      {
        fsVal ^= (msTab31l[(unsigned char)kmerSeq[i]][(k - 1 - i) % 31] | msTab33r[(unsigned char)kmerSeq[i]][(k - 1 - i) % 33]);
      }
    }
    hVal[j * m2] = fsVal;
    for (unsigned j2 = 1; j2 < m2; j2++)
    {
      uint64_t tVal = hVal[j * m2] * (j2 ^ k * multiSeed);
      tVal ^= tVal >> multiShift;
      hVal[j * m2 + j2] = tVal;
    }
  }
  for (unsigned int hIt = 0; hIt < m2; hIt++)
  {
    minhVal[hIt] = hVal[hIt * m];
    for (unsigned int seedIt = 1; seedIt < m; seedIt++)
    {
      minhVal[hIt] = MIN(minhVal[hIt], hVal[hIt * m + seedIt]);
    }
  }
}

// strand-aware multihash spaced seed ntHash with multiple hashes per seed
inline bool NTMSMC64(const char *kmerSeq, const std::vector<std::vector<unsigned>> &seedSeq, const unsigned k, const unsigned m, const unsigned m2,
                     uint64_t &fhVal, uint64_t &rhVal, unsigned &locN, uint64_t *hVal, uint64_t *minhVal, bool *hStn)
{
  fhVal = rhVal = 0;
  locN = 0;
  for (int i = k - 1; i >= 0; i--)
  {
    if (seedTab[(unsigned char)kmerSeq[i]] == seedN)
    {
      locN = i;
      return false;
    }
    fhVal = rol1(fhVal);
    fhVal = swapbits033(fhVal);
    fhVal ^= seedTab[(unsigned char)kmerSeq[k - 1 - i]];

    rhVal = rol1(rhVal);
    rhVal = swapbits033(rhVal);
    rhVal ^= seedTab[(unsigned char)kmerSeq[i] & cpOff];
  }

  for (unsigned j = 0; j < m; j++)
  {
    uint64_t fsVal = fhVal, rsVal = rhVal;
    for (unsigned i = 0; i < k; i++)
    {
      if (!seedSeq[j][i])
      {
        fsVal ^= (msTab31l[(unsigned char)kmerSeq[i]][(k - 1 - i) % 31] | msTab33r[(unsigned char)kmerSeq[i]][(k - 1 - i) % 33]);
        rsVal ^= (msTab31l[(unsigned char)kmerSeq[i] & cpOff][i % 31] | msTab33r[(unsigned char)kmerSeq[i] & cpOff][i % 33]);
      }
    }
    hStn[j * m2] = rsVal < fsVal;
    hVal[j * m2] = hStn[j * m2] ? rsVal : fsVal;
    for (unsigned j2 = 1; j2 < m2; j2++)
    {
      uint64_t tVal = hVal[j * m2] * (j2 ^ k * multiSeed);
      tVal ^= tVal >> multiShift;
      hStn[j * m2 + j2] = hStn[j * m2];
      hVal[j * m2 + j2] = tVal;
    }
  }
  for (unsigned int hIt = 0; hIt < m2; hIt++)
  {
    minhVal[hIt] = hVal[hIt * m];
    for (unsigned int seedIt = 1; seedIt < m; seedIt++)
    {
      minhVal[hIt] = MIN(minhVal[hIt], hVal[hIt * m + seedIt]);
    }
  }
  return true;
}

// strand-aware multihash spaced seed ntHash for sliding k-mers with multiple hashes per seed
inline void NTMSMC64(const char *kmerSeq, const std::vector<std::vector<unsigned>> &seedSeq, const unsigned char charOut, const unsigned char charIn,
                     const unsigned k, const unsigned m, const unsigned m2, uint64_t &fhVal, uint64_t &rhVal, uint64_t *hVal, uint64_t *minhVal, bool *hStn)
{
  fhVal = NTF64(fhVal, k, charOut, charIn);
  rhVal = NTR64(rhVal, k, charOut, charIn);
  for (unsigned j = 0; j < m; j++)
  {
    uint64_t fsVal = fhVal, rsVal = rhVal;
    for (unsigned i = 0; i < k; i++)
    {
      if (!seedSeq[j][i])
      {
        fsVal ^= (msTab31l[(unsigned char)kmerSeq[i]][(k - 1 - i) % 31] | msTab33r[(unsigned char)kmerSeq[i]][(k - 1 - i) % 33]);
        rsVal ^= (msTab31l[(unsigned char)kmerSeq[i] & cpOff][i % 31] | msTab33r[(unsigned char)kmerSeq[i] & cpOff][i % 33]);
      }
    }
    hStn[j * m2] = rsVal < fsVal;
    hVal[j * m2] = hStn[j * m2] ? rsVal : fsVal;
    for (unsigned j2 = 1; j2 < m2; j2++)
    {
      uint64_t tVal = hVal[j * m2] * (j2 ^ k * multiSeed);
      tVal ^= tVal >> multiShift;
      hStn[j * m2 + j2] = hStn[j * m2];
      hVal[j * m2 + j2] = tVal;
    }
  }
  for (unsigned int hIt = 0; hIt < m2; hIt++)
  {
    minhVal[hIt] = hVal[hIt * m];
    for (unsigned int seedIt = 1; seedIt < m; seedIt++)
    {
      minhVal[hIt] = MIN(minhVal[hIt], hVal[hIt * m + seedIt]);
    }
  }
}
