class StringDistance:
    def distance(self, s0, s1):
        raise NotImplementedError()


class NormalizedStringDistance(StringDistance):
    def distance(self, s0, s1):
        raise NotImplementedError()


class MetricStringDistance(StringDistance):
    def distance(self, s0, s1):
        raise NotImplementedError()


class StringSimilarity:
    def similarity(self, s0, s1):
        raise NotImplementedError()


class NormalizedStringSimilarity(StringSimilarity):
    def similarity(self, s0, s1):
        raise NotImplementedError()


class Levenshtein(MetricStringDistance):
    def distance(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 0.0
        if len(s0) == 0:
            return len(s1)
        if len(s1) == 0:
            return len(s0)

        v0 = [0] * (len(s1) + 1)
        v1 = [0] * (len(s1) + 1)

        for i in range(len(v0)):
            v0[i] = i

        for i in range(len(s0)):
            v1[0] = i + 1
            for j in range(len(s1)):
                cost = 1
                if s0[i] == s1[j]:
                    cost = 0
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            v0, v1 = v1, v0

        return v0[len(s1)]


class NormalizedLevenshtein(NormalizedStringDistance, NormalizedStringSimilarity):
    def __init__(self):
        self.levenshtein = Levenshtein()

    def distance(self, s0, s1):
        if s0 is None:
            raise TypeError("Argument s0 is NoneType.")
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s0 == s1:
            return 0.0
        m_len = max(len(s0), len(s1))
        if m_len == 0:
            return 0.0
        return self.levenshtein.distance(s0, s1) / m_len

    def similarity(self, s0, s1):
        return 1.0 - self.distance(s0, s1)
