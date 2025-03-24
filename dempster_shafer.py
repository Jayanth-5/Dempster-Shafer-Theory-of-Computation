from collections import defaultdict

class MassFunction:
    def __init__(self, frame, masses):
        """
        Initialize a mass function.
        
        Args:
            frame: Set of hypotheses (frame of discernment).
            masses: Dictionary mapping subsets (as frozensets) to their masses.
        """
        self.frame = frozenset(frame)
        self.masses = {frozenset(k): v for k, v in masses.items()}
        total_mass = sum(self.masses.values())
        if not abs(total_mass - 1.0) < 1e-6:
            raise ValueError(f"Sum of masses must be 1, got {total_mass}")

    def combine(self, other):
        """
        Combine this mass function with another using Dempster's rule.
        
        Args:
            other: Another MassFunction with the same frame.
        
        Returns:
            A new MassFunction representing the combined evidence.
        """
        if self.frame != other.frame:
            raise ValueError("Frames of discernment must match")
        
        combined_masses = defaultdict(float)
        K = 0  # Conflict factor
        for A in self.masses:
            for B in other.masses:
                C = A.intersection(B)
                if C:
                    combined_masses[C] += self.masses[A] * other.masses[B]
                else:
                    K += self.masses[A] * other.masses[B]
        
        if K == 1:
            raise ValueError("Total conflict, cannot combine")
        
        normalization = 1 / (1 - K)
        combined_masses = {C: v * normalization for C, v in combined_masses.items()}
        return MassFunction(self.frame, combined_masses)

    def belief(self, A):
        """
        Calculate the belief for a subset.
        
        Args:
            A: Subset (as a set or frozenset).
        
        Returns:
            float: Belief value for the subset.
        """
        A = frozenset(A)
        return sum(self.masses[B] for B in self.masses if B.issubset(A))

    def plausibility(self, A):
        """
        Calculate the plausibility for a subset.
        
        Args:
            A: Subset (as a set or frozenset).
        
        Returns:
            float: Plausibility value for the subset.
        """
        A = frozenset(A)
        return sum(self.masses[B] for B in self.masses if B.intersection(A))

    def __str__(self):
        """Return a string representation of the mass function."""
        lines = [f"Frame: {set(self.frame)}"]
        for subset, mass in self.masses.items():
            lines.append(f"m({set(subset)}) = {mass:.4f}")
        return "\n".join(lines)