# CTC temporal-order challenger

The existing temporal heads learn frame identity and derive action order afterward.  This
challenger instead uses each sequence answer as an ordered action transcript and trains with
CTC, marginalising unknown event boundaries.  HARn intervals and HAU action pools are auxiliary
losses.  At inference, all 24 answer permutations are scored by exact CTC likelihood.

The first gate is fold 0 under the repository's subject-disjoint split.  Scale-out is allowed
only if exact order, pairwise order, or residual complementarity beats the current temporal
evidence.
